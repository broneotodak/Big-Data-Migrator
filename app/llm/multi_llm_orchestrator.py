"""
Multi-LLM Orchestrator for consensus-based responses.

This module orchestrates multiple LLM providers to provide the best possible
responses by running queries in parallel and comparing results.
"""
import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.utils.logging_config import get_logger
from app.memory.memory_monitor import MemoryMonitor
from app.llm.lm_studio_client import LMStudioClient
from app.llm.anthropic_client import AnthropicClient, AnthropicConfig
from app.llm.online_llm_fallback import OnlineLLMFallback, OnlineLLMConfig

logger = get_logger(__name__)

@dataclass
class LLMResponse:
    """Response from a single LLM provider."""
    provider: str
    response: str
    response_time_ms: float
    success: bool
    error: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MultiLLMResult:
    """Result from multiple LLM providers."""
    responses: List[LLMResponse]
    best_response: Optional[LLMResponse]
    consensus_response: Optional[str]
    total_time_ms: float
    providers_used: List[str]
    metadata: Dict[str, Any]

class ResponseQualityEvaluator:
    """Evaluates and scores LLM responses for quality."""
    
    def __init__(self):
        self.quality_metrics = [
            "relevance", "completeness", "accuracy", "clarity", "actionability"
        ]
    
    def evaluate_response(self, response: str, query: str, context: Optional[Dict] = None) -> float:
        """
        Evaluate response quality on a scale of 0-1.
        
        Args:
            response: The LLM response to evaluate
            query: Original user query
            context: Optional context information
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        metrics_count = 0
        
        # Length appropriateness (not too short, not too verbose)
        length_score = self._evaluate_length(response, query)
        score += length_score
        metrics_count += 1
        
        # Structure and formatting
        structure_score = self._evaluate_structure(response)
        score += structure_score
        metrics_count += 1
        
        # Relevance to query
        relevance_score = self._evaluate_relevance(response, query)
        score += relevance_score
        metrics_count += 1
        
        # Presence of actionable insights
        actionability_score = self._evaluate_actionability(response)
        score += actionability_score
        metrics_count += 1
        
        # Data analysis quality (if applicable)
        if context and "data" in str(context).lower():
            data_analysis_score = self._evaluate_data_analysis(response)
            score += data_analysis_score
            metrics_count += 1
        
        return score / metrics_count if metrics_count > 0 else 0.0
    
    def _evaluate_length(self, response: str, query: str) -> float:
        """Evaluate if response length is appropriate."""
        response_length = len(response)
        query_length = len(query)
        
        # Simple heuristic: response should be 2-20x query length for most cases
        expected_min = query_length * 2
        expected_max = query_length * 20
        
        if expected_min <= response_length <= expected_max:
            return 1.0
        elif response_length < expected_min:
            return max(0.3, response_length / expected_min)
        else:
            return max(0.5, expected_max / response_length)
    
    def _evaluate_structure(self, response: str) -> float:
        """Evaluate response structure and formatting."""
        score = 0.0
        
        # Check for bullet points or numbered lists
        if any(marker in response for marker in ['•', '-', '*', '1.', '2.', '3.']):
            score += 0.3
        
        # Check for clear sections or paragraphs
        if '\n\n' in response or response.count('\n') >= 2:
            score += 0.3
        
        # Check for questions or recommendations
        if any(word in response.lower() for word in ['recommend', 'suggest', 'consider', 'try']):
            score += 0.2
        
        # Check for data-specific language
        if any(word in response.lower() for word in ['data', 'column', 'row', 'table', 'analysis']):
            score += 0.2
        
        return min(1.0, score)
    
    def _evaluate_relevance(self, response: str, query: str) -> float:
        """Evaluate how relevant the response is to the query."""
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(response_words))
        relevance = overlap / len(query_words) if query_words else 0
        
        return min(1.0, relevance * 2)  # Scale up relevance score
    
    def _evaluate_actionability(self, response: str) -> float:
        """Evaluate if response provides actionable insights."""
        actionable_indicators = [
            'should', 'could', 'recommend', 'suggest', 'try', 'consider',
            'next step', 'action', 'implement', 'apply', 'use', 'create'
        ]
        
        found_indicators = sum(1 for indicator in actionable_indicators 
                             if indicator in response.lower())
        
        return min(1.0, found_indicators / 3)  # Normalize to 0-1
    
    def _evaluate_data_analysis(self, response: str) -> float:
        """Evaluate quality of data analysis in response."""
        data_terms = [
            'column', 'row', 'table', 'dataset', 'relationship', 'pattern',
            'trend', 'correlation', 'distribution', 'outlier', 'quality',
            'schema', 'index', 'foreign key', 'primary key'
        ]
        
        found_terms = sum(1 for term in data_terms if term in response.lower())
        return min(1.0, found_terms / 5)  # Normalize to 0-1

class MultiLLMOrchestrator:
    """
    Orchestrates multiple LLM providers for consensus-based responses.
    
    This class:
    - Runs queries against multiple LLMs in parallel
    - Evaluates response quality
    - Provides best response selection
    - Offers consensus building
    """
    
    def __init__(self,
                 local_llm_client: Optional[LMStudioClient] = None,
                 anthropic_client: Optional[AnthropicClient] = None,
                 openai_client: Optional[OnlineLLMFallback] = None,
                 memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize the Multi-LLM Orchestrator.
        
        Args:
            local_llm_client: Local LM Studio client
            anthropic_client: Anthropic Claude client
            openai_client: OpenAI client
            memory_monitor: Memory monitoring instance
        """
        self.local_llm = local_llm_client
        self.anthropic = anthropic_client
        self.openai = openai_client
        self.memory_monitor = memory_monitor or MemoryMonitor()
        
        self.evaluator = ResponseQualityEvaluator()
        self.response_cache = {}
        
        # Track which LLMs are available
        self.available_llms = {}
        if self.local_llm:
            self.available_llms["local"] = self.local_llm
        if self.anthropic:
            self.available_llms["anthropic"] = self.anthropic
        if self.openai:
            self.available_llms["openai"] = self.openai
        
        logger.info(f"MultiLLMOrchestrator initialized with {len(self.available_llms)} providers: {list(self.available_llms.keys())}")
    
    async def get_multi_llm_response(self, 
                                   query: str,
                                   context: Optional[Dict[str, Any]] = None,
                                   providers: Optional[List[str]] = None,
                                   include_consensus: bool = True) -> MultiLLMResult:
        """
        Get responses from multiple LLM providers.
        
        Args:
            query: User query
            context: Optional context information
            providers: Specific providers to use (None for all available)
            include_consensus: Whether to build consensus response
            
        Returns:
            Multi-LLM result with all responses and best selection
        """
        start_time = time.time()
        
        # Determine which providers to use
        if providers is None:
            providers = list(self.available_llms.keys())
        else:
            providers = [p for p in providers if p in self.available_llms]
        
        if not providers:
            raise ValueError("No LLM providers available")
        
        # Run queries in parallel
        responses = await self._run_parallel_queries(query, context, providers)
        
        # Evaluate response quality
        for response in responses:
            if response.success:
                response.confidence_score = self.evaluator.evaluate_response(
                    response.response, query, context
                )
        
        # Select best response
        best_response = self._select_best_response(responses)
        
        # Build consensus if requested
        consensus_response = None
        if include_consensus and len([r for r in responses if r.success]) > 1:
            consensus_response = await self._build_consensus(responses, query, context)
        
        total_time = (time.time() - start_time) * 1000
        
        return MultiLLMResult(
            responses=responses,
            best_response=best_response,
            consensus_response=consensus_response,
            total_time_ms=total_time,
            providers_used=providers,
            metadata={
                "query_length": len(query),
                "successful_responses": len([r for r in responses if r.success]),
                "average_confidence": sum(r.confidence_score or 0 for r in responses if r.success) / max(1, len([r for r in responses if r.success]))
            }
        )
    
    async def _run_parallel_queries(self, 
                                  query: str, 
                                  context: Optional[Dict[str, Any]], 
                                  providers: List[str]) -> List[LLMResponse]:
        """Run queries against multiple providers in parallel."""
        
        def run_query(provider_name: str) -> LLMResponse:
            start_time = time.time()
            
            try:
                if provider_name == "local" and self.local_llm:
                    # For local LLM, use simple chat completion
                    response = self.local_llm.chat_completion([{"role": "user", "content": query}])
                    
                elif provider_name == "anthropic" and self.anthropic:
                    # For Anthropic, use data analysis if context available
                    if context:
                        response = self.anthropic.analyze_data_structure(context, query)
                    else:
                        response = self.anthropic.chat_completion([{"role": "user", "content": query}])
                    
                elif provider_name == "openai" and self.openai:
                    # For OpenAI, use chat completion
                    response = self.openai.chat_completion([{"role": "user", "content": query}])
                    
                else:
                    raise ValueError(f"Provider {provider_name} not available")
                
                response_time = (time.time() - start_time) * 1000
                
                return LLMResponse(
                    provider=provider_name,
                    response=response,
                    response_time_ms=response_time,
                    success=True
                )
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                logger.error(f"Error with {provider_name}: {str(e)}")
                
                return LLMResponse(
                    provider=provider_name,
                    response="",
                    response_time_ms=response_time,
                    success=False,
                    error=str(e)
                )
        
        # Execute queries in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(providers)) as executor:
            futures = {executor.submit(run_query, provider): provider for provider in providers}
            responses = []
            
            for future in as_completed(futures):
                response = future.result()
                responses.append(response)
        
        return responses
    
    def _select_best_response(self, responses: List[LLMResponse]) -> Optional[LLMResponse]:
        """Select the best response based on quality metrics."""
        successful_responses = [r for r in responses if r.success and r.confidence_score is not None]
        
        if not successful_responses:
            return None
        
        # Sort by confidence score (highest first)
        successful_responses.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return successful_responses[0]
    
    async def _build_consensus(self, 
                             responses: List[LLMResponse], 
                             query: str, 
                             context: Optional[Dict[str, Any]]) -> str:
        """Build a consensus response from multiple LLM outputs."""
        successful_responses = [r for r in responses if r.success]
        
        # If only one successful response, return it as the consensus
        if len(successful_responses) == 1:
            return f"Single provider response from {successful_responses[0].provider}:\n\n{successful_responses[0].response}"
        
        # If no successful responses, return appropriate message
        if len(successful_responses) == 0:
            return "No successful responses from available LLM providers."
        
        # Use the best available LLM to build consensus
        consensus_builder = None
        if self.anthropic:
            consensus_builder = self.anthropic
        elif self.openai:
            consensus_builder = self.openai
        elif self.local_llm:
            consensus_builder = self.local_llm
        
        if not consensus_builder:
            # If no consensus builder available, return the best response
            best_response = max(successful_responses, key=lambda x: x.confidence_score or 0)
            return f"Best available response from {best_response.provider}:\n\n{best_response.response}"
        
        # Create consensus prompt
        responses_text = "\n\n".join([
            f"Response from {r.provider} (confidence: {r.confidence_score:.2f}):\n{r.response}"
            for r in successful_responses
        ])
        
        consensus_prompt = f"""I need you to create a consensus response based on multiple AI system outputs for this query: "{query}"

Here are the different responses:

{responses_text}

Please create a unified, comprehensive response that:
1. Combines the best insights from each response
2. Resolves any contradictions by favoring higher-confidence answers
3. Provides a clear, actionable final answer
4. Maintains the technical accuracy of the original responses

Unified Response:"""
        
        try:
            if hasattr(consensus_builder, 'chat_completion'):
                consensus = consensus_builder.chat_completion([{"role": "user", "content": consensus_prompt}])
            else:
                consensus = "Multiple perspectives available - see individual responses above."
            
            return consensus
            
        except Exception as e:
            logger.error(f"Error building consensus: {str(e)}")
            # Fallback to best response if consensus building fails
            best_response = max(successful_responses, key=lambda x: x.confidence_score or 0)
            return f"Consensus building failed. Best response from {best_response.provider}:\n\n{best_response.response}"
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about available providers."""
        stats = {}
        
        for provider_name, client in self.available_llms.items():
            try:
                if hasattr(client, 'check_connection'):
                    status = client.check_connection()
                else:
                    status = {"status": "unknown"}
                
                stats[provider_name] = {
                    "available": True,
                    "status": status.get("status", "unknown"),
                    "model": getattr(client, 'model', 'unknown')
                }
                
            except Exception as e:
                stats[provider_name] = {
                    "available": False,
                    "error": str(e),
                    "model": "unknown"
                }
        
        return stats
    
    def get_response_comparison(self, multi_result: MultiLLMResult) -> Dict[str, Any]:
        """Generate a detailed comparison of responses."""
        comparison = {
            "summary": {
                "total_providers": len(multi_result.providers_used),
                "successful_responses": len([r for r in multi_result.responses if r.success]),
                "average_response_time": sum(r.response_time_ms for r in multi_result.responses) / len(multi_result.responses),
                "best_provider": multi_result.best_response.provider if multi_result.best_response else None
            },
            "individual_results": [],
            "consensus_available": multi_result.consensus_response is not None
        }
        
        for response in multi_result.responses:
            result_info = {
                "provider": response.provider,
                "success": response.success,
                "response_time_ms": response.response_time_ms,
                "response_length": len(response.response) if response.success else 0,
                "confidence_score": response.confidence_score,
                "error": response.error
            }
            comparison["individual_results"].append(result_info)
        
        return comparison 