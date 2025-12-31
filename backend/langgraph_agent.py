"""
langgraph_agent.py - LangGraph-based ReAct agent with multi-collection support
Implements reasoning-action-observation loop for intelligent tool selection
"""

import warnings
warnings.filterwarnings("ignore")

from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from backend.query import get_llm, get_agent_system_prompt
from backend.tools import get_all_tools
import json

# Try to import Langfuse for tracing
try:
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
    print("✓ Langfuse 3.x LangChain CallbackHandler loaded")
except ImportError:
    CallbackHandler = None
    LANGFUSE_AVAILABLE = False
    print("⚠ Langfuse LangChain integration not available")


# ============================================================================
# AGENT STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """
    The state of the agent throughout its execution.
    LangGraph maintains this state across all nodes.
    
    Fields:
        messages: Conversation history (HumanMessage, AIMessage, ToolMessage)
        intermediate_steps: Track tool calls and observations for debugging
        tool_call_history: Track which tools have been called to prevent loops
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    intermediate_steps: Annotated[list, operator.add]
    tool_call_history: Annotated[list, operator.add]


# ============================================================================
# LANGGRAPH REACT AGENT
# ============================================================================

class LangGraphReActAgent:
    """
    LangGraph-based ReAct agent that intelligently routes queries to tools.
    
    Flow:
        User Query → Reasoning → Action (Tool Call) → Observation → 
        → Reasoning → [More Actions OR Final Answer]
    
    The agent automatically selects tools based on their descriptions.
    """
    
    def __init__(self):
        """Initialize the agent with LLM and tools"""
        self.llm = get_llm()
        self.tools = get_all_tools()
        
        # Bind tools to LLM (enables function calling)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create tool node for executing tools
        self.tool_node = ToolNode(self.tools)
        
        # Build the state graph
        self.graph = self._build_graph()
        
        print(f"[AGENT] ✓ Initialized LangGraph ReAct Agent with {len(self.tools)} tools")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine for ReAct pattern"""
        
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Define nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("action", self._action_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "action",
                "end": END
            }
        )
        
        # After action, always go back to agent for reasoning
        workflow.add_edge("action", "agent")
        
        # Compile the graph
        return workflow.compile()
    
    def _agent_node(self, state: AgentState) -> AgentState:
        """
        Agent reasoning node: Decides what action to take next.
        The LLM with tools bound will either:
        1. Call a tool (if it needs more information)
        2. Provide a final answer (if it has enough information)
        """
        messages = state["messages"]
        intermediate_steps = state.get("intermediate_steps", [])
        tool_history = state.get("tool_call_history", [])
        
        # SAFETY: If limit reached, force final answer (no tool binding)
        if len(intermediate_steps) >= 5:
            force_msg = HumanMessage(
                content="""You have used the maximum number of tools (5). Now provide ONLY your final answer to the user's original question.

IMPORTANT: 
- Do NOT explain what tools you used
- Do NOT describe your reasoning process  
- ONLY provide the clear, direct answer to the question
- Format it as a complete, helpful response"""
            )
            messages = list(messages) + [force_msg]
            response = self.llm.invoke(messages)  # No tool binding
            print("[AGENT] Forced final answer - limit reached (5 tools)")
            return {"messages": [response]}
        
        # SAFETY: If repeated tool detected, force final answer
        if len(tool_history) > len(set(tool_history)):
            force_msg = HumanMessage(
                content="""You've already used this tool. Based on the results you have, provide ONLY your final answer to the user's question.

IMPORTANT:
- Do NOT explain your reasoning
- ONLY provide the clear, direct answer
- Format it as a complete, helpful response"""
            )
            messages = list(messages) + [force_msg]
            response = self.llm.invoke(messages)  # No tool binding
            print("[AGENT] Forced final answer - repeated tool detected")
            return {"messages": [response]}
        
        # Normal operation: Invoke LLM with tool binding
        response = self.llm_with_tools.invoke(messages)
        
        # Log if response has content (reasoning)
        if hasattr(response, 'content') and response.content:
            print(f"[AGENT THOUGHT] {response.content[:200]}...")
        
        # Log if response has tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                print(f"[AGENT ACTION] Calling {tc.get('name')}...")
        
        return {"messages": [response]}
    
    def _action_node(self, state: AgentState) -> AgentState:
        """
        Action execution node: Executes the tool selected by the agent.
        Uses ToolNode to handle tool execution automatically.
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # Extract tool calls for logging
        tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
        
        intermediate_steps = []
        tool_names = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("args", {})
            tool_names.append(tool_name)
            
            print(f"\n[ACTION] Executing tool: {tool_name}")
            print(f"[ACTION] Arguments: {tool_args}")
            
            intermediate_steps.append({
                "tool": tool_name,
                "input": tool_args,
                "output": "executing..."  # Will be filled after execution
            })
        
        # Use ToolNode to execute all tool calls automatically
        # ToolNode handles the message creation and tool execution
        result = self.tool_node.invoke(state)
        
        # Log results
        if "messages" in result:
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    print(f"[OBSERVATION] Result length: {len(msg.content)} chars")
        
        # Add intermediate steps and tool history to result
        result["intermediate_steps"] = intermediate_steps
        result["tool_call_history"] = tool_names
        
        return result
    
    def _should_continue(self, state: AgentState) -> str:
        """
        Routing function: Decides whether to continue with more actions or end.
        This enforces stopping conditions at the routing level.
        
        Returns:
            "continue": Agent wants to call tools (and limits not exceeded)
            "end": Agent provided answer OR limits reached
        """
        messages = state["messages"]
        last_message = messages[-1]
        intermediate_steps = state.get("intermediate_steps", [])
        tool_history = state.get("tool_call_history", [])
        
        # Check if the last message has tool calls
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
        
        # STOP CONDITION 1: No tool calls = agent provided final answer
        if not has_tool_calls:
            print("[ROUTING] No tool calls - agent provided final answer - ending")
            return "end"
        
        # STOP CONDITION 2: Maximum tool calls reached (max 5)
        if len(intermediate_steps) >= 5:
            print("[ROUTING] Maximum tool calls (5) reached - will force answer on next agent cycle")
            return "end"
        
        # STOP CONDITION 3: Repeated tool calls detected
        if len(tool_history) > len(set(tool_history)):
            print("[ROUTING] Repeated tool call detected - will force answer on next agent cycle")
            return "end"
        
        # All checks passed - continue to action node
        print(f"[ROUTING] Agent wants to call tool - continuing to action node")
        return "continue"
    
    def invoke(self, query: str, max_iterations: int = 5, langfuse_handler=None) -> dict:
        """
        Run the agent on a query with optional Langfuse tracing.
        
        Args:
            query: User's question
            max_iterations: Maximum reasoning-action loops (prevents infinite loops)
            langfuse_handler: Optional Langfuse callback handler for tracing
        
        Returns:
            Dictionary with answer and execution trace
        """
        print(f"\n{'='*80}")
        print(f"[AGENT] Processing query: {query}")
        if langfuse_handler:
            print(f"[AGENT] Langfuse tracing enabled")
        print(f"{'='*80}\n")
        
        # Get system prompt
        system_prompt = get_agent_system_prompt()
        
        # Initialize state with system prompt + user message
        initial_state = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ],
            "intermediate_steps": [],
            "tool_call_history": []
        }
        
        # Run the graph
        try:
            # Build config with Langfuse callback if provided
            config = {"recursion_limit": 25}
            if langfuse_handler:
                config["callbacks"] = [langfuse_handler]
            
            final_state = self.graph.invoke(initial_state, config)
            
            # Extract final answer
            messages = final_state["messages"]
            final_message = messages[-1]
            
            # Get the answer (last AI message without tool calls)
            # This is the clean final answer, not the reasoning
            answer = final_message.content
            
            # If answer is empty or None, try to find last non-empty AI response
            if not answer or not answer.strip():
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
                        answer = msg.content
                        break
            
            # Extract intermediate steps for transparency
            steps = final_state.get("intermediate_steps", [])
            
            print(f"\n[AGENT] ✓ Completed with {len(steps)} tool calls\n")
            print(f"[AGENT] Final answer preview: {answer[:200] if answer else 'No answer'}...")
            
            # Build detailed reasoning trace from messages
            reasoning_trace = self._extract_reasoning_trace(messages)
            
            # Serialize messages for response
            serialized_messages = []
            for msg in messages:
                try:
                    if hasattr(msg, 'model_dump'):
                        serialized_messages.append(msg.model_dump())
                    elif hasattr(msg, 'dict'):
                        serialized_messages.append(msg.dict())
                    else:
                        serialized_messages.append(str(msg))
                except Exception:
                    serialized_messages.append(str(msg))
            
            return {
                "answer": answer,
                "intermediate_steps": steps,
                "messages": serialized_messages,
                "tools_used": [step["tool"] for step in steps],
                "reasoning_trace": reasoning_trace
            }
            
        except Exception as e:
            print(f"\n[AGENT] ✗ Error during execution: {e}\n")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "intermediate_steps": [],
                "messages": [],
                "tools_used": [],
                "reasoning_trace": [],
                "error": str(e)
            }
    
    def _extract_reasoning_trace(self, messages: list) -> list:
        """
        Extract detailed reasoning trace from message history.
        
        Returns list of dicts with:
        - type: 'thought' | 'action' | 'observation'
        - content: The actual content
        - tool_name: (for actions only)
        - tool_args: (for actions only)
        """
        trace = []
        
        for i, msg in enumerate(messages):
            # Skip system messages
            if isinstance(msg, SystemMessage):
                continue
            
            # Skip the initial user query
            if isinstance(msg, HumanMessage) and i <= 1:
                continue
            
            # Skip our internal reasoning prompts
            if isinstance(msg, HumanMessage) and ("Before taking action" in msg.content or 
                                                   "Good. Now call the appropriate tool" in msg.content or
                                                   "Based on the tool results" in msg.content):
                continue
            
            # AI messages with reasoning
            if isinstance(msg, AIMessage):
                # Check if this message has tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # If there's text content, that's the thought/reasoning
                    if msg.content and msg.content.strip():
                        trace.append({
                            "type": "thought",
                            "content": msg.content
                        })
                    
                    # Add each tool call as an action
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                        
                        # Format the action with detail
                        action_text = f"**Calling Tool:** `{tool_name}`"
                        if tool_args:
                            if 'query' in tool_args:
                                action_text += f"\n**Search Query:** \"{tool_args['query']}\""
                            if 'k' in tool_args:
                                action_text += f"\n**Documents to retrieve:** {tool_args['k']}"
                            # Add any other args
                            for k, v in tool_args.items():
                                if k not in ['query', 'k']:
                                    action_text += f"\n**{k}:** {v}"
                        
                        trace.append({
                            "type": "action",
                            "content": action_text,
                            "tool_name": tool_name,
                            "tool_args": tool_args
                        })
                else:
                    # Final answer or intermediate reasoning without tool calls
                    if msg.content and msg.content.strip():
                        trace.append({
                            "type": "thought",
                            "content": msg.content
                        })
            
            # Tool messages (observations)
            elif isinstance(msg, ToolMessage):
                # Truncate long observations for readability
                content = msg.content
                original_length = len(content)
                
                if len(content) > 800:
                    content = content[:800] + f"\n\n... _(truncated {original_length - 800} characters)_"
                
                trace.append({
                    "type": "observation",
                    "content": content
                })
        
        return trace
        
        return trace
    
    async def astream(self, query: str, max_iterations: int = 5):
        """
        Stream the agent's execution (for real-time UI updates).
        
        Args:
            query: User's question
            max_iterations: Maximum reasoning-action loops
        
        Yields:
            Dictionary chunks with partial results
        """
        print(f"\n[AGENT] Streaming query: {query}\n")
        
        # Get system prompt
        system_prompt = get_agent_system_prompt()
        
        # Initialize state with system prompt + user message
        initial_state = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ],
            "intermediate_steps": [],
            "tool_call_history": []
        }
        
        # Stream the graph execution
        try:
            async for state_update in self.graph.astream(
                initial_state,
                {"recursion_limit": max_iterations}
            ):
                # Yield intermediate updates
                yield {
                    "type": "state_update",
                    "data": state_update
                }
            
            print(f"[AGENT] ✓ Streaming completed\n")
            
        except Exception as e:
            print(f"[AGENT] ✗ Streaming error: {e}\n")
            yield {
                "type": "error",
                "error": str(e)
            }


# ============================================================================
# GLOBAL AGENT INSTANCE
# ============================================================================

_agent = None

def get_agent() -> LangGraphReActAgent:
    """Get or create the shared agent instance"""
    global _agent
    if _agent is None:
        _agent = LangGraphReActAgent()
    return _agent


def query_agent(question: str) -> dict:
    """
    Convenience function to query the agent.
    
    Args:
        question: User's query
    
    Returns:
        Dictionary with answer and metadata
    """
    agent = get_agent()
    return agent.invoke(question)
