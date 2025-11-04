#!/usr/bin/env python3
"""
Cost-Benefit Analyzer - LLM Improvement ROI Tracking
Analyzes the cost-benefit ratio of LLM corrections and tracks success metrics.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CostBenefitMetrics:
    """Cost-benefit metrics for LLM correction analysis."""
    total_llm_cost: float
    total_accuracy_improvement: float
    cost_per_accuracy_point: float
    processing_time_increase: float
    roi_percentage: float
    meets_cost_threshold: bool
    meets_time_threshold: bool
    overall_success: bool

@dataclass
class LLMCostBreakdown:
    """Breakdown of LLM processing costs."""
    total_requests: int
    total_tokens: int
    cost_per_token: float
    total_processing_time: float
    cost_per_second: float
    cost_per_request: float

class CostBenefitAnalyzer:
    """Analyzes cost-benefit ratio of LLM corrections."""
    
    def __init__(self, 
                 cost_per_token: float = 0.0001,
                 cost_per_second: float = 0.01,
                 max_cost_per_accuracy_point: float = 0.10,
                 max_time_increase: float = 0.5):
        """
        Initialize Cost-Benefit Analyzer.
        
        Args:
            cost_per_token: Cost per token for LLM processing
            cost_per_second: Cost per second of processing time
            max_cost_per_accuracy_point: Maximum cost per 1% accuracy improvement
            max_time_increase: Maximum acceptable time increase (as fraction)
        """
        self.cost_per_token = cost_per_token
        self.cost_per_second = cost_per_second
        self.max_cost_per_accuracy_point = max_cost_per_accuracy_point
        self.max_time_increase = max_time_increase
        
        # Tracking data
        self.llm_costs: List[Dict[str, Any]] = []
        self.accuracy_improvements: List[Dict[str, Any]] = []
        self.processing_times: List[Dict[str, Any]] = []
    
    def track_llm_cost(self, 
                      request_id: str,
                      tokens_used: int,
                      processing_time: float,
                      model: str = "unknown") -> None:
        """
        Track LLM processing cost.
        
        Args:
            request_id: Unique identifier for the request
            tokens_used: Number of tokens used
            processing_time: Processing time in seconds
            model: LLM model used
        """
        cost_data = {
            'request_id': request_id,
            'tokens_used': tokens_used,
            'processing_time': processing_time,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'token_cost': tokens_used * self.cost_per_token,
            'time_cost': processing_time * self.cost_per_second,
            'total_cost': (tokens_used * self.cost_per_token) + (processing_time * self.cost_per_second)
        }
        
        self.llm_costs.append(cost_data)
        logger.debug(f"Tracked LLM cost: {cost_data['total_cost']:.4f} for request {request_id}")
    
    def track_accuracy_improvement(self,
                                 document_id: str,
                                 page_number: int,
                                 before_accuracy: float,
                                 after_accuracy: float,
                                 improvement: float) -> None:
        """
        Track accuracy improvement from LLM correction.
        
        Args:
            document_id: Document identifier
            page_number: Page number
            before_accuracy: Accuracy before LLM correction
            after_accuracy: Accuracy after LLM correction
            improvement: Improvement percentage
        """
        improvement_data = {
            'document_id': document_id,
            'page_number': page_number,
            'before_accuracy': before_accuracy,
            'after_accuracy': after_accuracy,
            'improvement': improvement,
            'timestamp': datetime.now().isoformat()
        }
        
        self.accuracy_improvements.append(improvement_data)
        logger.debug(f"Tracked accuracy improvement: {improvement:.1%} for {document_id} page {page_number}")
    
    def track_processing_time(self,
                            document_id: str,
                            page_number: int,
                            before_time: float,
                            after_time: float,
                            time_increase: float) -> None:
        """
        Track processing time impact of LLM correction.
        
        Args:
            document_id: Document identifier
            page_number: Page number
            before_time: Processing time before LLM
            after_time: Processing time after LLM
            time_increase: Time increase as fraction
        """
        time_data = {
            'document_id': document_id,
            'page_number': page_number,
            'before_time': before_time,
            'after_time': after_time,
            'time_increase': time_increase,
            'timestamp': datetime.now().isoformat()
        }
        
        self.processing_times.append(time_data)
        logger.debug(f"Tracked time increase: {time_increase:.1%} for {document_id} page {page_number}")
    
    def calculate_cost_benefit_metrics(self) -> CostBenefitMetrics:
        """Calculate overall cost-benefit metrics."""
        if not self.llm_costs or not self.accuracy_improvements:
            return CostBenefitMetrics(
                total_llm_cost=0.0,
                total_accuracy_improvement=0.0,
                cost_per_accuracy_point=0.0,
                processing_time_increase=0.0,
                roi_percentage=0.0,
                meets_cost_threshold=True,
                meets_time_threshold=True,
                overall_success=True
            )
        
        # Calculate total LLM cost
        total_llm_cost = sum(cost['total_cost'] for cost in self.llm_costs)
        
        # Calculate total accuracy improvement
        total_accuracy_improvement = sum(imp['improvement'] for imp in self.accuracy_improvements)
        
        # Calculate cost per accuracy point
        cost_per_accuracy_point = 0.0
        if total_accuracy_improvement > 0:
            cost_per_accuracy_point = total_llm_cost / (total_accuracy_improvement * 100)
        
        # Calculate processing time increase
        avg_time_increase = 0.0
        if self.processing_times:
            avg_time_increase = sum(time['time_increase'] for time in self.processing_times) / len(self.processing_times)
        
        # Calculate ROI
        roi_percentage = 0.0
        if total_llm_cost > 0 and total_accuracy_improvement > 0:
            # Simple ROI: accuracy improvement per dollar spent
            roi_percentage = (total_accuracy_improvement * 100) / total_llm_cost
        
        # Check thresholds
        meets_cost_threshold = cost_per_accuracy_point <= self.max_cost_per_accuracy_point
        meets_time_threshold = avg_time_increase <= self.max_time_increase
        overall_success = meets_cost_threshold and meets_time_threshold
        
        return CostBenefitMetrics(
            total_llm_cost=total_llm_cost,
            total_accuracy_improvement=total_accuracy_improvement,
            cost_per_accuracy_point=cost_per_accuracy_point,
            processing_time_increase=avg_time_increase,
            roi_percentage=roi_percentage,
            meets_cost_threshold=meets_cost_threshold,
            meets_time_threshold=meets_time_threshold,
            overall_success=overall_success
        )
    
    def get_llm_cost_breakdown(self) -> LLMCostBreakdown:
        """Get detailed breakdown of LLM costs."""
        if not self.llm_costs:
            return LLMCostBreakdown(
                total_requests=0,
                total_tokens=0,
                cost_per_token=self.cost_per_token,
                total_processing_time=0.0,
                cost_per_second=self.cost_per_second,
                cost_per_request=0.0
            )
        
        total_requests = len(self.llm_costs)
        total_tokens = sum(cost['tokens_used'] for cost in self.llm_costs)
        total_processing_time = sum(cost['processing_time'] for cost in self.llm_costs)
        total_cost = sum(cost['total_cost'] for cost in self.llm_costs)
        
        return LLMCostBreakdown(
            total_requests=total_requests,
            total_tokens=total_tokens,
            cost_per_token=self.cost_per_token,
            total_processing_time=total_processing_time,
            cost_per_second=self.cost_per_second,
            cost_per_request=total_cost / total_requests if total_requests > 0 else 0.0
        )
    
    def get_success_metrics(self) -> Dict[str, Any]:
        """Get success metrics for LLM correction."""
        cost_benefit = self.calculate_cost_benefit_metrics()
        cost_breakdown = self.get_llm_cost_breakdown()
        
        return {
            'cost_benefit_metrics': {
                'total_llm_cost': cost_benefit.total_llm_cost,
                'total_accuracy_improvement': cost_benefit.total_accuracy_improvement,
                'cost_per_accuracy_point': cost_benefit.cost_per_accuracy_point,
                'processing_time_increase': cost_benefit.processing_time_increase,
                'roi_percentage': cost_benefit.roi_percentage,
                'meets_cost_threshold': cost_benefit.meets_cost_threshold,
                'meets_time_threshold': cost_benefit.meets_time_threshold,
                'overall_success': cost_benefit.overall_success
            },
            'cost_breakdown': {
                'total_requests': cost_breakdown.total_requests,
                'total_tokens': cost_breakdown.total_tokens,
                'total_processing_time': cost_breakdown.total_processing_time,
                'cost_per_request': cost_breakdown.cost_per_request
            },
            'thresholds': {
                'max_cost_per_accuracy_point': self.max_cost_per_accuracy_point,
                'max_time_increase': self.max_time_increase
            },
            'summary': {
                'total_measurements': len(self.accuracy_improvements),
                'average_improvement': cost_benefit.total_accuracy_improvement / len(self.accuracy_improvements) if self.accuracy_improvements else 0.0,
                'success_rate': 1.0 if cost_benefit.overall_success else 0.0
            }
        }
    
    def export_analysis(self, output_file: str) -> None:
        """Export cost-benefit analysis to JSON file."""
        try:
            analysis_data = {
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'analyzer_version': '1.0',
                    'cost_per_token': self.cost_per_token,
                    'cost_per_second': self.cost_per_second
                },
                'success_metrics': self.get_success_metrics(),
                'llm_costs': self.llm_costs,
                'accuracy_improvements': self.accuracy_improvements,
                'processing_times': self.processing_times
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported cost-benefit analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export cost-benefit analysis: {e}")
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on cost-benefit analysis."""
        recommendations = []
        cost_benefit = self.calculate_cost_benefit_metrics()
        
        if not cost_benefit.overall_success:
            if not cost_benefit.meets_cost_threshold:
                recommendations.append(
                    f"⚠️ **High cost per accuracy point**: ${cost_benefit.cost_per_accuracy_point:.4f} "
                    f"(threshold: ${self.max_cost_per_accuracy_point:.2f}) - Consider optimizing LLM prompts or using cheaper models"
                )
            
            if not cost_benefit.meets_time_threshold:
                recommendations.append(
                    f"⚠️ **High processing time increase**: {cost_benefit.processing_time_increase:.1%} "
                    f"(threshold: {self.max_time_increase:.1%}) - Consider reducing LLM processing or using faster models"
                )
        else:
            recommendations.append("✅ **Good cost-benefit ratio** - LLM correction is providing value within acceptable thresholds")
        
        if cost_benefit.roi_percentage > 100:
            recommendations.append(f"🎉 **Excellent ROI**: {cost_benefit.roi_percentage:.1f}% - LLM correction is highly cost-effective")
        elif cost_benefit.roi_percentage > 50:
            recommendations.append(f"✅ **Good ROI**: {cost_benefit.roi_percentage:.1f}% - LLM correction is cost-effective")
        else:
            recommendations.append(f"⚠️ **Low ROI**: {cost_benefit.roi_percentage:.1f}% - Consider optimizing LLM correction strategy")
        
        return recommendations

# Convenience functions
def create_cost_benefit_analyzer(cost_per_token: float = 0.0001,
                               cost_per_second: float = 0.01,
                               max_cost_per_accuracy_point: float = 0.10,
                               max_time_increase: float = 0.5) -> CostBenefitAnalyzer:
    """Create a new Cost-Benefit Analyzer instance."""
    return CostBenefitAnalyzer(cost_per_token, cost_per_second, max_cost_per_accuracy_point, max_time_increase)





