"""
agent.py - Agent routing and state management
Handles intelligent routing between Qdrant DB and web search (ReAct framework)
"""

import warnings
warnings.filterwarnings("ignore")
from typing import Dict, Any
import uuid
from backend.query import load_qdrant_vectordb, get_llm, create_rag_chain
from backend.tools import web_search_tool

# Try to import Langfuse for tracing (optional)
# Langfuse temporarily disabled - package version incompatibility
# Reinstall: pip install "langfuse>=2.40.0,<3.0"
LANGFUSE_AVAILABLE = False
langfuse_client = None
print("[AGENT] ⚠ Langfuse temporarily disabled - agent will work without tracing")


def react_agent_qdrant(query: str, k: int = 5) -> Dict[str, Any]:
    """
    ReAct agent for Qdrant + Web Search routing.
    Uses reasoning to decide: Try Qdrant first, fallback to web if insufficient.
    
    Args:
        query: User's question
        k: Number of chunks to retrieve from Qdrant
    
    Returns:
        Dictionary with answer, source, thought process, and retrieved data
    """
    print(f"\n[REACT AGENT] Processing query: {query}")
    thought_process = []
    llm = get_llm()
    
    # === LANGFUSE TRACING SETUP ===
    # Create a unique trace ID for this agent execution
    trace_id = str(uuid.uuid4())
    trace = None
    langfuse_handler = None
    
    if LANGFUSE_AVAILABLE and langfuse_client:
        try:
            # Create a trace for the entire agent execution
            # This will show up in Langfuse dashboard as a single conversation
            trace = langfuse_client.trace(
                id=trace_id,
                name="react_agent_query",
                input={"query": query, "k": k},
                metadata={
                    "agent_type": "ReAct",
                    "tools": ["qdrant_search", "web_search"],
                    "model": "llama-3.1-8b-instant"
                }
            )
            # Get handler for LangChain integration
            langfuse_handler = trace.get_langchain_handler()
            print(f"[LANGFUSE] ✓ Trace created: {trace_id}")
        except Exception as e:
            print(f"[LANGFUSE] ⚠ Failed to create trace: {e}")
            trace = None
            langfuse_handler = None
    
    # STEP 1: Think - Analyze the query and decide strategy
    print("Step 1: Analyzing query...")
    
    # === LANGFUSE: Track Step 1 as a span ===
    if trace:
        try:
            step1_span = langfuse_client.span(
                trace_id=trace_id,
                name="step_1_initial_analysis",
                input={"query": query},
                metadata={"step": 1, "action": "Think"}
            )
        except:
            step1_span = None
    
    initial_thought_prompt = f"""You are a ReAct agent analyzing a user query. Explain your reasoning about what the user is asking and your strategy.

User Query: {query}

Think step by step:
1. What is the user asking about?
2. Should I search my internal knowledge base first, or go straight to web search?
3. What kind of information am I looking for?

Provide your reasoning in 2-3 sentences."""

    # Invoke LLM with Langfuse callback (if available)
    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    initial_thought = llm.invoke(initial_thought_prompt, config=config).content
    thought_process.append(f"**Initial Analysis:** {initial_thought}")
    print(f"Thought: {initial_thought[:100]}...")
    
    # Update Step 1 span with output
    if trace and 'step1_span' in locals() and step1_span:
        try:
            langfuse_client.span(
                id=step1_span.id,
                output={"reasoning": initial_thought}
            )
        except:
            pass
    
    # STEP 2: Act - Try Qdrant first
    thought_process.append("**Action:** Searching Qdrant knowledge base for relevant documents...")
    print("Step 2: Searching Qdrant database...")
    
    # === LANGFUSE: Track Step 2 - Qdrant Search ===
    if trace:
        try:
            step2_span = langfuse_client.span(
                trace_id=trace_id,
                name="step_2_qdrant_search",
                input={"query": query, "k": k},
                metadata={"step": 2, "action": "Act", "tool": "qdrant"}
            )
        except:
            step2_span = None
    
    try:
        # Load Qdrant and search
        qdrant_db = load_qdrant_vectordb()
        retriever = qdrant_db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        
        # STEP 3: Observe - Evaluate results
        if docs and len(docs) > 0:
            # Check if results are relevant (simple heuristic: check if any doc has reasonable similarity)
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            thought_process.append(f"**Observation:** Retrieved {len(docs)} chunks from Qdrant knowledge base.")
            print(f"✓ Found {len(docs)} relevant chunks")
            
            # Update Step 2 span with results
            if trace and 'step2_span' in locals() and step2_span:
                try:
                    langfuse_client.span(
                        id=step2_span.id,
                        output={"num_docs": len(docs), "success": True}
                    )
                except:
                    pass
            
            # STEP 4: Think - Evaluate if results are sufficient
            # === LANGFUSE: Track Step 4 - Evaluation ===
            if trace:
                try:
                    step4_span = langfuse_client.span(
                        trace_id=trace_id,
                        name="step_4_evaluation",
                        input={"query": query, "context_preview": context[:200]},
                        metadata={"step": 4, "action": "Think", "num_chunks": len(docs)}
                    )
                except:
                    step4_span = None
            
            eval_prompt = f"""You are a ReAct agent evaluating Qdrant knowledge base search results. Decide if the retrieved context can answer the user's query.

User Query: {query}

Retrieved Context from Qdrant DB (first 500 chars):
{context[:500]}...

Full context has {len(docs)} document chunks available.

IMPORTANT: Be conservative - if the context contains ANY relevant information about the query topic, use it. Only reject if the context is completely unrelated or empty.

Evaluate:
1. Does this context discuss the topic asked about?
2. Can I provide a useful answer from this context (even if partial)?
3. Is the context completely irrelevant or should I use it?

Respond in this exact format:
DECISION: [USE_QDRANT or NEED_WEB]
REASONING: [Your 1-2 sentence explanation]"""

            # Invoke evaluation with Langfuse tracking
            config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
            evaluation_thought = llm.invoke(eval_prompt, config=config).content
            thought_process.append(f"**Evaluation:** {evaluation_thought}")
            print(f"Evaluation: {evaluation_thought[:100]}...")
            
            # Update Step 4 span
            if trace and 'step4_span' in locals() and step4_span:
                try:
                    langfuse_client.span(
                        id=step4_span.id,
                        output={"decision": evaluation_thought[:500]}
                    )
                except:
                    pass
            
            # Improved decision logic - look for explicit decision
            use_qdrant = False
            if "USE_QDRANT" in evaluation_thought.upper():
                use_qdrant = True
            elif "NEED_WEB" in evaluation_thought.upper():
                use_qdrant = False
            else:
                # Fallback: if no explicit decision, check sentiment
                positive_keywords = ["can answer", "sufficient", "relevant", "contains information", "discusses", "addresses"]
                negative_keywords = ["not relevant", "insufficient", "unrelated", "doesn't contain", "no information"]
                
                positive_count = sum(1 for kw in positive_keywords if kw in evaluation_thought.lower())
                negative_count = sum(1 for kw in negative_keywords if kw in evaluation_thought.lower())
                
                # Prefer using Qdrant if unclear (conservative approach)
                use_qdrant = positive_count >= negative_count
            
            if use_qdrant:
                thought_process.append("**Decision:** Context from Qdrant knowledge base is relevant. Generating answer from retrieved documents.")
                print("✓ Using Qdrant context to generate answer...")
                
                # === LANGFUSE: Track Step 5 - Answer Generation ===
                if trace:
                    try:
                        step5_span = langfuse_client.span(
                            trace_id=trace_id,
                            name="step_5_answer_generation",
                            input={"query": query, "source": "qdrant"},
                            metadata={"step": 5, "action": "Generate", "k": k}
                        )
                    except:
                        step5_span = None
                
                # Generate answer using RAG (with Langfuse callback)
                rag_chain, _ = create_rag_chain(qdrant_db, llm, k=k)
                config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
                answer = rag_chain.invoke(query, config=config)
                
                # Check if RAG chain found the context actually useful
                if "[NO_CONTEXT_FOUND]" in answer:
                    thought_process.append("**Re-evaluation:** RAG chain determined context was not actually useful. Proceeding to web search.")
                    print("⚠ RAG returned [NO_CONTEXT_FOUND], falling back to web search...")
                    
                    # Update span to show fallback
                    if trace and 'step5_span' in locals() and step5_span:
                        try:
                            langfuse_client.span(
                                id=step5_span.id,
                                output={"fallback": True, "reason": "NO_CONTEXT_FOUND"}
                            )
                        except:
                            pass
                    # Continue to web search below (don't return here)
                else:
                    # Update span with successful generation
                    if trace and 'step5_span' in locals() and step5_span:
                        try:
                            langfuse_client.span(
                                id=step5_span.id,
                                output={"answer_length": len(answer), "source": "qdrant"}
                            )
                        except:
                            pass
                    
                    # Update main trace with final output
                    if trace:
                        try:
                            langfuse_client.trace(
                                id=trace_id,
                                output={
                                    "answer": answer,
                                    "source": "qdrant",
                                    "num_chunks": len(docs)
                                }
                            )
                            langfuse_client.flush()  # Ensure data is sent
                            print(f"[LANGFUSE] ✓ Trace completed: {trace_id}")
                        except Exception as e:
                            print(f"[LANGFUSE] ⚠ Failed to update trace: {e}")
                    
                    return {
                        "answer": answer,
                        "source": "qdrant",
                        "thought_process": thought_process,
                        "chunks": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
                        "web_results": None
                    }
            else:
                thought_process.append("**Decision:** Retrieved context is not relevant to the query. Proceeding to web search for current information.")
                print("⚠ Context not relevant, trying web search...")
        else:
            thought_process.append("**Observation:** No documents found in Qdrant knowledge base.")
            thought_process.append("**Decision:** Need to search the web for information.")
            print("⚠ No results in Qdrant, trying web search...")
            
    except Exception as e:
        thought_process.append(f"**Observation:** Qdrant search failed with error: {str(e)}")
        thought_process.append("**Decision:** Falling back to web search due to error.")
        print(f"⚠ Qdrant error: {e}")
    
    # STEP 5: Think - Plan web search
    print("Step 3: Planning web search...")
    
    # === LANGFUSE: Track Web Search Planning ===
    if trace:
        try:
            web_plan_span = langfuse_client.span(
                trace_id=trace_id,
                name="step_5_web_search_planning",
                input={"query": query},
                metadata={"step": 5, "action": "Think", "reason": "qdrant_insufficient"}
            )
        except:
            web_plan_span = None
    
    web_search_thought_prompt = f"""You are a ReAct agent planning a web search. The internal knowledge base didn't have the information needed.

User Query: {query}

Think about:
1. Why might the internal knowledge base not have this information?
2. What should I search for on the web to get the best results?
3. What type of answer am I expecting to find?

Provide your reasoning in 2-3 sentences."""

    config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
    web_search_thought = llm.invoke(web_search_thought_prompt, config=config).content
    thought_process.append(f"**Planning Web Search:** {web_search_thought}")
    print(f"Planning: {web_search_thought[:100]}...")
    
    # Update planning span
    if trace and 'web_plan_span' in locals() and web_plan_span:
        try:
            langfuse_client.span(
                id=web_plan_span.id,
                output={"reasoning": web_search_thought}
            )
        except:
            pass
    
    # STEP 6: Act - Web search
    thought_process.append("**Action:** Executing web search via Tavily...")
    print("Step 4: Searching the web...")
    
    # === LANGFUSE: Track Web Search Execution ===
    if trace:
        try:
            web_search_span = langfuse_client.span(
                trace_id=trace_id,
                name="step_6_web_search",
                input={"query": query, "num_results": 5},
                metadata={"step": 6, "action": "Act", "tool": "tavily"}
            )
        except:
            web_search_span = None
    
    # STEP 6: Act - Web search
    try:
        web_results = web_search_tool(query, num_results=5)
        
        # Update web search span with results
        if trace and 'web_search_span' in locals() and web_search_span:
            try:
                langfuse_client.span(
                    id=web_search_span.id,
                    output={"num_results": len(web_results) if web_results else 0, "success": True}
                )
            except:
                pass
        
        if web_results:
            # STEP 7: Observe web results
            thought_process.append(f"**Observation:** Found {len(web_results)} web results from Tavily.")
            print(f"✓ Found {len(web_results)} web results")
            
            # Generate answer from web results
            web_context = "\n\n".join([
                f"Source: {r['title']}\n{r['snippet']}" 
                for r in web_results[:3]
            ])
            
            # STEP 8: Think - Synthesize final answer
            # === LANGFUSE: Track Final Synthesis ===
            if trace:
                try:
                    synthesis_span = langfuse_client.span(
                        trace_id=trace_id,
                        name="step_7_synthesis_planning",
                        input={"query": query, "num_web_results": len(web_results)},
                        metadata={"step": 7, "action": "Think"}
                    )
                except:
                    synthesis_span = None
            
            synthesis_prompt = f"""You are a ReAct agent preparing to synthesize a final answer from web search results.

User Query: {query}

Web Results Summary (first 3):
{web_context[:500]}...

Think about:
1. What are the key facts from these web results?
2. How should I structure my answer?
3. What's the best way to present this information to the user?

Provide your reasoning in 2-3 sentences."""

            config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
            synthesis_thought = llm.invoke(synthesis_prompt, config=config).content
            thought_process.append(f"**Final Synthesis:** {synthesis_thought}")
            print(f"Synthesizing: {synthesis_thought[:100]}...")
            
            # Update synthesis span
            if trace and 'synthesis_span' in locals() and synthesis_span:
                try:
                    langfuse_client.span(
                        id=synthesis_span.id,
                        output={"reasoning": synthesis_thought}
                    )
                except:
                    pass
            
            thought_process.append("**Action:** Generating comprehensive answer from web sources.")
            
            # === LANGFUSE: Track Final Answer Generation ===
            if trace:
                try:
                    final_gen_span = langfuse_client.span(
                        trace_id=trace_id,
                        name="step_8_final_answer_generation",
                        input={"query": query, "source": "web"},
                        metadata={"step": 8, "action": "Generate"}
                    )
                except:
                    final_gen_span = None
            
            # Use LLM to synthesize answer from web results
            web_prompt = f"""Based on the following web search results, provide a comprehensive answer to the user's question.

Question: {query}

Web Search Results:
{web_context}

Provide a clear, well-structured answer based on the information above. If the information is insufficient, say so."""

            config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
            answer = llm.invoke(web_prompt, config=config).content
            
            # Update final generation span
            if trace and 'final_gen_span' in locals() and final_gen_span:
                try:
                    langfuse_client.span(
                        id=final_gen_span.id,
                        output={"answer_length": len(answer)}
                    )
                except:
                    pass
            
            # Update main trace with final output
            if trace:
                try:
                    langfuse_client.trace(
                        id=trace_id,
                        output={
                            "answer": answer,
                            "source": "web",
                            "num_web_results": len(web_results)
                        }
                    )
                    langfuse_client.flush()  # Ensure data is sent to Langfuse
                    print(f"[LANGFUSE] ✓ Trace completed: {trace_id}")
                except Exception as e:
                    print(f"[LANGFUSE] ⚠ Failed to update trace: {e}")
            
            return {
                "answer": answer,
                "source": "web",
                "thought_process": thought_process,
                "chunks": None,
                "web_results": web_results
            }
        else:
            thought_process.append("**Observation:** Web search returned no results.")
            print("⚠ No web results found")
            
    except Exception as e:
        thought_process.append(f"**Observation:** Web search failed with error: {str(e)}")
        print(f"❌ Web search error: {e}")
        
        # Update trace with error info
        if trace:
            try:
                langfuse_client.trace(
                    id=trace_id,
                    output={"error": str(e), "source": "none"}
                )
                langfuse_client.flush()
                print(f"[LANGFUSE] ✓ Trace updated with error: {trace_id}")
            except Exception as trace_error:
                print(f"[LANGFUSE] ⚠ Failed to update trace with error: {trace_error}")
    
    # STEP 9: Final fallback - No information found
    thought_process.append("**Final Decision:** Unable to find relevant information from either knowledge base or web search. Apologizing to user.")
    
    # Update trace with final fallback
    if trace:
        try:
            langfuse_client.trace(
                id=trace_id,
                output={"answer": "No information found", "source": "none", "fallback": True}
            )
            langfuse_client.flush()
            print(f"[LANGFUSE] ✓ Trace completed with fallback: {trace_id}")
        except Exception as e:
            print(f"[LANGFUSE] ⚠ Failed to update trace: {e}")
    
    return {
        "answer": "I apologize, but I couldn't find relevant information to answer your question. Both the knowledge base and web search were unable to provide sufficient context.",
        "source": "none",
        "thought_process": thought_process,
        "chunks": None,
        "web_results": None
    }
