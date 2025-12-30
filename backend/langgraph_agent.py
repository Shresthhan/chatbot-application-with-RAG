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
        This is where the LLM analyzes the query and decides which tool to call.
        """
        messages = state["messages"]
        intermediate_steps = state.get("intermediate_steps", [])
        
        # Check if we've hit the tool limit and need to force a final answer
        if len(intermediate_steps) >= 5:
            # Add a message instructing the agent to provide final answer
            force_answer_msg = HumanMessage(
                content="You have used 5 tools. Based on the information gathered, provide your final answer now. Do NOT call any more tools."
            )
            messages = list(messages) + [force_answer_msg]
            
            # Invoke LLM WITHOUT tool binding to force text response
            response = self.llm.invoke(messages)
            print("[AGENT] Forced final answer generation (no more tool calls allowed)")
        else:
            # Normal operation: Invoke LLM with tool binding
            response = self.llm_with_tools.invoke(messages)
        
        # Return updated state
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
        
        Returns:
            "continue": Agent wants to call more tools
            "end": Agent has sufficient information to answer
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # Check if the last message has tool calls
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
        
        # HARD STOP 1: Check step limit (max 5 tool calls)
        # But only stop if we're NOT about to execute a tool (i.e., we're at reasoning stage)
        intermediate_steps = state.get("intermediate_steps", [])
        if len(intermediate_steps) >= 5 and not has_tool_calls:
            print("[STOP CONDITION] Maximum 5 tool calls reached. Agent must provide final answer.")
            return "end"
        
        # HARD STOP 2: Check for repeated tool calls
        tool_history = state.get("tool_call_history", [])
        if len(tool_history) > len(set(tool_history)) and not has_tool_calls:
            print("[STOP CONDITION] Repeated tool call detected. Agent must provide final answer.")
            return "end"
        
        # If the last message has tool calls, continue to action
        if has_tool_calls:
            # But check if we're about to exceed the limit
            if len(intermediate_steps) >= 5:
                print("[STOP CONDITION] Tool call requested but limit reached. Blocking tool execution.")
                return "end"
            return "continue"
        
        # Otherwise, we're done (agent provided final answer)
        return "end"
    
    def invoke(self, query: str, max_iterations: int = 5) -> dict:
        """
        Run the agent on a query.
        
        Args:
            query: User's question
            max_iterations: Maximum reasoning-action loops (prevents infinite loops)
        
        Returns:
            Dictionary with answer and execution trace
        """
        print(f"\n{'='*80}")
        print(f"[AGENT] Processing query: {query}")
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
            final_state = self.graph.invoke(
                initial_state,
                {"recursion_limit": 25}  # Increased from 5 to 25
            )
            
            # Extract final answer
            messages = final_state["messages"]
            final_message = messages[-1]
            
            # Get the answer (last AI message without tool calls)
            answer = final_message.content
            
            # Extract intermediate steps for transparency
            steps = final_state.get("intermediate_steps", [])
            
            print(f"\n[AGENT] ✓ Completed with {len(steps)} tool calls\n")
            
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
                "tools_used": [step["tool"] for step in steps]
            }
            
        except Exception as e:
            print(f"\n[AGENT] ✗ Error during execution: {e}\n")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "intermediate_steps": [],
                "messages": [],
                "tools_used": [],
                "error": str(e)
            }
    
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
