"""
Ensemble Optimization and Dynamic Weighting Module

This module provides advanced ensemble optimization capabilities including
dynamic weight adjustment, diversity validation, and performance monitoring.
"""

import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import json
import os

from src.forecasting_models import BaseForecaster, EnsembleForecaster


# Configure logging
logger = logging.getLogger('ensemble_optimizer')


@dataclass
class EnsemblePerformance:
    """Data class for tracking ensemble performance metrics."""
    ensemble_name: str
    ticker: str
    prediction_accuracy: float
    model_diversity: float
    weight_stability: float
    computation_time: float
    individual_model_scores: Dict[str, float]
    ensemble_weights: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class DiversityMetrics:
    """Data class for ensemble diversity measurements."""
    correlation_matrix: Dict[str, Dict[str, float]]
    average_correlation: float
    diversity_score: float  # 1 - average_correlation
    model_disagreement: float
    prediction_variance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class EnsembleOptimizer:
    """
    Advanced ensemble optimization system with dynamic weighting and performance monitoring.
    
    This class provides sophisticated ensemble management including:
    - Dynamic weight adjustment based on recent performance
    - Diversity requirements and validation
    - Performance monitoring and reporting
    - Automatic model selection and removal
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ensemble optimizer.
        
        Args:
            config: Configuration dictionary for optimizer settings
        """
        self.logger = logging.getLogger('ensemble_optimizer')
        
        # Default configuration
        self.config = {
            'performance_window': 30,  # Days to consider for recent performance
            'weight_update_frequency': 5,  # Update weights every N predictions
            'diversity_threshold': 0.3,  # Minimum diversity score required
            'min_correlation_threshold': 0.8,  # Maximum correlation between models
            'performance_decay': 0.95,  # Decay factor for historical performance
            'min_model_weight': 0.05,  # Minimum weight for any model
            'max_model_weight': 0.5,  # Maximum weight for any single model
            'underperformer_threshold': 0.1,  # Remove models below this performance
            'stability_window': 10,  # Window for weight stability calculation
            'monitoring_enabled': True,
            'auto_rebalance': True,
            'diversity_penalty': 0.1  # Penalty for low diversity
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # State tracking
        self.performance_history = {}  # Dict[ensemble_name, List[EnsemblePerformance]]
        self.weight_history = {}  # Dict[ensemble_name, List[Dict[str, float]]]
        self.diversity_history = {}  # Dict[ensemble_name, List[DiversityMetrics]]
        self.prediction_count = {}  # Dict[ensemble_name, int]
        self.last_optimization = {}  # Dict[ensemble_name, datetime]
        
    def optimize_ensemble(self, 
                         ensemble: EnsembleForecaster,
                         recent_data: pd.Series,
                         performance_metrics: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """
        Optimize ensemble weights and composition based on recent performance.
        
        Args:
            ensemble: EnsembleForecaster to optimize
            recent_data: Recent time series data for validation
            performance_metrics: Optional recent performance metrics for each model
            
        Returns:
            Dictionary containing optimization results and recommendations
        """
        try:
            ensemble_name = ensemble.model_name
            self.logger.info(f"Starting ensemble optimization for {ensemble_name}")
            
            # Calculate current diversity metrics
            diversity_metrics = self._calculate_diversity_metrics(ensemble, recent_data)
            
            # Update performance tracking
            if performance_metrics:
                self._update_performance_tracking(ensemble_name, performance_metrics)
            
            # Calculate optimal weights
            optimal_weights = self._calculate_optimal_weights(ensemble, recent_data, diversity_metrics)
            
            # Check for underperforming models
            underperformers = self._identify_underperformers(ensemble, performance_metrics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                ensemble, diversity_metrics, optimal_weights, underperformers
            )
            
            # Apply optimizations if auto-rebalance is enabled
            if self.config.get('auto_rebalance', True):
                self._apply_optimizations(ensemble, optimal_weights, underperformers)
            
            # Record optimization results
            optimization_result = {
                'ensemble_name': ensemble_name,
                'timestamp': datetime.now(),
                'diversity_metrics': diversity_metrics.to_dict(),
                'optimal_weights': optimal_weights,
                'current_weights': ensemble._model_weights.copy(),
                'underperformers': underperformers,
                'recommendations': recommendations,
                'optimization_applied': self.config.get('auto_rebalance', True)
            }
            
            self.last_optimization[ensemble_name] = datetime.now()
            
            self.logger.info(f"Ensemble optimization completed for {ensemble_name}")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Ensemble optimization failed: {e}")
            raise RuntimeError(f"Ensemble optimization failed: {e}")
    
    def monitor_ensemble_performance(self, 
                                   ensemble: EnsembleForecaster,
                                   actual_values: List[float],
                                   predicted_values: List[float],
                                   individual_predictions: Dict[str, List[float]]) -> EnsemblePerformance:
        """
        Monitor and record ensemble performance metrics.
        
        Args:
            ensemble: EnsembleForecaster being monitored
            actual_values: Actual observed values
            predicted_values: Ensemble predictions
            individual_predictions: Individual model predictions
            
        Returns:
            EnsemblePerformance object with calculated metrics
        """
        try:
            ensemble_name = ensemble.model_name
            
            # Calculate prediction accuracy
            accuracy = self._calculate_prediction_accuracy(actual_values, predicted_values)
            
            # Calculate individual model scores
            individual_scores = {}
            for model_name, predictions in individual_predictions.items():
                if len(predictions) == len(actual_values):
                    individual_scores[model_name] = self._calculate_prediction_accuracy(
                        actual_values, predictions
                    )
            
            # Calculate diversity metrics
            diversity_score = self._calculate_prediction_diversity(individual_predictions)
            
            # Calculate weight stability
            weight_stability = self._calculate_weight_stability(ensemble_name, ensemble._model_weights)
            
            # Create performance record
            performance = EnsemblePerformance(
                ensemble_name=ensemble_name,
                ticker="portfolio",  # Could be made more specific
                prediction_accuracy=accuracy,
                model_diversity=diversity_score,
                weight_stability=weight_stability,
                computation_time=0.0,  # Would need to be measured externally
                individual_model_scores=individual_scores,
                ensemble_weights=ensemble._model_weights.copy(),
                timestamp=datetime.now()
            )
            
            # Store performance history
            if ensemble_name not in self.performance_history:
                self.performance_history[ensemble_name] = []
            
            self.performance_history[ensemble_name].append(performance)
            
            # Keep only recent history
            max_history = self.config.get('performance_window', 30)
            if len(self.performance_history[ensemble_name]) > max_history:
                self.performance_history[ensemble_name] = self.performance_history[ensemble_name][-max_history:]
            
            self.logger.info(f"Performance recorded for {ensemble_name}: accuracy={accuracy:.4f}, diversity={diversity_score:.4f}")
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            raise RuntimeError(f"Performance monitoring failed: {e}")
    
    def _calculate_diversity_metrics(self, ensemble: EnsembleForecaster, data: pd.Series) -> DiversityMetrics:
        """
        Calculate diversity metrics for ensemble models.
        
        Args:
            ensemble: EnsembleForecaster to analyze
            data: Data for generating predictions to measure diversity
            
        Returns:
            DiversityMetrics object
        """
        try:
            # Generate predictions from each model for diversity calculation
            model_predictions = {}
            
            # Use a subset of data for efficiency
            test_data = data[-min(100, len(data)):]  # Last 100 points or all data
            
            for model_name, model in ensemble._base_models.items():
                try:
                    predictions = []
                    for i in range(min(20, len(test_data))):  # Generate 20 predictions
                        pred = model.predict(periods=1)
                        predictions.append(pred)
                    model_predictions[model_name] = predictions
                except Exception as e:
                    self.logger.warning(f"Could not generate predictions for diversity calculation from {model_name}: {e}")
                    continue
            
            if len(model_predictions) < 2:
                # Not enough models for diversity calculation
                return DiversityMetrics(
                    correlation_matrix={},
                    average_correlation=0.0,
                    diversity_score=1.0,
                    model_disagreement=0.0,
                    prediction_variance=0.0
                )
            
            # Calculate correlation matrix
            correlation_matrix = {}
            correlations = []
            
            model_names = list(model_predictions.keys())
            for i, model1 in enumerate(model_names):
                correlation_matrix[model1] = {}
                for j, model2 in enumerate(model_names):
                    if i == j:
                        correlation_matrix[model1][model2] = 1.0
                    elif j > i:  # Calculate only upper triangle
                        try:
                            corr = np.corrcoef(model_predictions[model1], model_predictions[model2])[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                            correlation_matrix[model1][model2] = corr
                            correlation_matrix[model2] = correlation_matrix.get(model2, {})
                            correlation_matrix[model2][model1] = corr
                            correlations.append(abs(corr))
                        except Exception:
                            correlation_matrix[model1][model2] = 0.0
                            correlation_matrix[model2] = correlation_matrix.get(model2, {})
                            correlation_matrix[model2][model1] = 0.0
            
            # Calculate average correlation and diversity
            avg_correlation = np.mean(correlations) if correlations else 0.0
            diversity_score = 1.0 - avg_correlation
            
            # Calculate model disagreement (standard deviation of predictions)
            all_predictions = np.array(list(model_predictions.values()))
            if all_predictions.size > 0:
                prediction_variance = np.var(all_predictions, axis=0).mean()
                model_disagreement = np.std(all_predictions, axis=0).mean()
            else:
                prediction_variance = 0.0
                model_disagreement = 0.0
            
            return DiversityMetrics(
                correlation_matrix=correlation_matrix,
                average_correlation=avg_correlation,
                diversity_score=diversity_score,
                model_disagreement=model_disagreement,
                prediction_variance=prediction_variance
            )
            
        except Exception as e:
            self.logger.error(f"Diversity calculation failed: {e}")
            # Return default metrics
            return DiversityMetrics(
                correlation_matrix={},
                average_correlation=0.0,
                diversity_score=1.0,
                model_disagreement=0.0,
                prediction_variance=0.0
            )
    
    def _calculate_optimal_weights(self, 
                                 ensemble: EnsembleForecaster,
                                 data: pd.Series,
                                 diversity_metrics: DiversityMetrics) -> Dict[str, float]:
        """
        Calculate optimal weights considering both performance and diversity.
        
        Args:
            ensemble: EnsembleForecaster to optimize
            data: Recent data for validation
            diversity_metrics: Current diversity metrics
            
        Returns:
            Dictionary of optimal weights for each model
        """
        try:
            # Get current performance scores
            performance_scores = {}
            
            for model_name, model in ensemble._base_models.items():
                try:
                    # Validate model on recent data
                    validation_result = model.validate(data, test_size=0.3)
                    mape = validation_result.get('mape', 100.0)
                    
                    # Convert MAPE to performance score (lower MAPE = higher score)
                    performance_score = 1.0 / (1.0 + mape / 100.0)
                    performance_scores[model_name] = performance_score
                    
                except Exception as e:
                    self.logger.warning(f"Performance calculation failed for {model_name}: {e}")
                    performance_scores[model_name] = 0.1  # Low score for failed models
            
            # Apply diversity penalty for highly correlated models
            diversity_adjusted_scores = performance_scores.copy()
            
            if diversity_metrics.correlation_matrix:
                diversity_penalty = self.config.get('diversity_penalty', 0.1)
                
                for model1 in performance_scores:
                    penalty = 0.0
                    for model2 in performance_scores:
                        if model1 != model2 and model1 in diversity_metrics.correlation_matrix:
                            correlation = diversity_metrics.correlation_matrix[model1].get(model2, 0.0)
                            if abs(correlation) > self.config.get('min_correlation_threshold', 0.8):
                                penalty += diversity_penalty * abs(correlation)
                    
                    diversity_adjusted_scores[model1] = max(0.01, performance_scores[model1] - penalty)
            
            # Calculate weights from adjusted scores
            total_score = sum(diversity_adjusted_scores.values())
            optimal_weights = {}
            
            if total_score > 0:
                for model_name, score in diversity_adjusted_scores.items():
                    weight = score / total_score
                    
                    # Apply weight constraints
                    weight = max(weight, self.config.get('min_model_weight', 0.05))
                    weight = min(weight, self.config.get('max_model_weight', 0.5))
                    
                    optimal_weights[model_name] = weight
            else:
                # Fallback to equal weights
                equal_weight = 1.0 / len(performance_scores)
                for model_name in performance_scores:
                    optimal_weights[model_name] = equal_weight
            
            # Normalize weights to sum to 1
            total_weight = sum(optimal_weights.values())
            if total_weight > 0:
                for model_name in optimal_weights:
                    optimal_weights[model_name] /= total_weight
            
            return optimal_weights
            
        except Exception as e:
            self.logger.error(f"Optimal weight calculation failed: {e}")
            # Return current weights as fallback
            return ensemble._model_weights.copy()
    
    def _identify_underperformers(self, 
                                ensemble: EnsembleForecaster,
                                performance_metrics: Optional[Dict[str, List[float]]]) -> List[str]:
        """
        Identify models that are consistently underperforming.
        
        Args:
            ensemble: EnsembleForecaster to analyze
            performance_metrics: Recent performance metrics
            
        Returns:
            List of underperforming model names
        """
        underperformers = []
        threshold = self.config.get('underperformer_threshold', 0.1)
        
        try:
            if performance_metrics:
                for model_name, scores in performance_metrics.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        if avg_score < threshold:
                            underperformers.append(model_name)
                            self.logger.info(f"Identified underperformer: {model_name} (avg score: {avg_score:.4f})")
            
            # Also check historical performance if available
            ensemble_name = ensemble.model_name
            if ensemble_name in self.performance_history:
                recent_performances = self.performance_history[ensemble_name][-10:]  # Last 10 records
                
                for model_name in ensemble._base_models.keys():
                    model_scores = []
                    for perf in recent_performances:
                        if model_name in perf.individual_model_scores:
                            model_scores.append(perf.individual_model_scores[model_name])
                    
                    if model_scores:
                        avg_score = sum(model_scores) / len(model_scores)
                        if avg_score < threshold and model_name not in underperformers:
                            underperformers.append(model_name)
                            self.logger.info(f"Identified historical underperformer: {model_name} (avg score: {avg_score:.4f})")
            
        except Exception as e:
            self.logger.error(f"Underperformer identification failed: {e}")
        
        return underperformers
    
    def _generate_recommendations(self, 
                                ensemble: EnsembleForecaster,
                                diversity_metrics: DiversityMetrics,
                                optimal_weights: Dict[str, float],
                                underperformers: List[str]) -> List[str]:
        """
        Generate optimization recommendations based on analysis.
        
        Args:
            ensemble: EnsembleForecaster being analyzed
            diversity_metrics: Current diversity metrics
            optimal_weights: Calculated optimal weights
            underperformers: List of underperforming models
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            # Diversity recommendations
            if diversity_metrics.diversity_score < self.config.get('diversity_threshold', 0.3):
                recommendations.append(
                    f"Low ensemble diversity detected (score: {diversity_metrics.diversity_score:.3f}). "
                    "Consider adding models with different approaches."
                )
            
            # High correlation recommendations
            if diversity_metrics.correlation_matrix:
                high_corr_pairs = []
                for model1, correlations in diversity_metrics.correlation_matrix.items():
                    for model2, corr in correlations.items():
                        if model1 < model2 and abs(corr) > self.config.get('min_correlation_threshold', 0.8):
                            high_corr_pairs.append((model1, model2, corr))
                
                if high_corr_pairs:
                    recommendations.append(
                        f"High correlation detected between models: {high_corr_pairs[:3]}. "
                        "Consider removing one model from each highly correlated pair."
                    )
            
            # Weight adjustment recommendations
            current_weights = ensemble._model_weights
            significant_changes = []
            
            for model_name in optimal_weights:
                if model_name in current_weights:
                    weight_change = abs(optimal_weights[model_name] - current_weights[model_name])
                    if weight_change > 0.1:  # Significant change threshold
                        significant_changes.append((model_name, weight_change))
            
            if significant_changes:
                recommendations.append(
                    f"Significant weight adjustments recommended for: {[name for name, _ in significant_changes[:3]]}"
                )
            
            # Underperformer recommendations
            if underperformers:
                recommendations.append(
                    f"Consider removing underperforming models: {underperformers}"
                )
            
            # Model count recommendations
            model_count = len(ensemble._base_models)
            min_models = self.config.get('min_models', 3)
            max_models = self.config.get('max_models', 7)
            
            if model_count < min_models:
                recommendations.append(
                    f"Ensemble has only {model_count} models. Consider adding more for better robustness."
                )
            elif model_count > max_models:
                recommendations.append(
                    f"Ensemble has {model_count} models. Consider reducing for better efficiency."
                )
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis error.")
        
        return recommendations
    
    def _apply_optimizations(self, 
                           ensemble: EnsembleForecaster,
                           optimal_weights: Dict[str, float],
                           underperformers: List[str]) -> None:
        """
        Apply optimization recommendations to the ensemble.
        
        Args:
            ensemble: EnsembleForecaster to optimize
            optimal_weights: Optimal weights to apply
            underperformers: Models to potentially remove
        """
        try:
            # Update weights
            for model_name, weight in optimal_weights.items():
                if model_name in ensemble._model_weights:
                    ensemble._model_weights[model_name] = weight
            
            # Remove underperformers if we have enough models remaining
            remaining_models = len(ensemble._base_models) - len(underperformers)
            min_models = self.config.get('min_models', 3)
            
            if remaining_models >= min_models:
                for model_name in underperformers:
                    if model_name in ensemble._base_models:
                        del ensemble._base_models[model_name]
                        if model_name in ensemble._model_weights:
                            del ensemble._model_weights[model_name]
                        self.logger.info(f"Removed underperforming model: {model_name}")
                
                # Renormalize weights after removal
                total_weight = sum(ensemble._model_weights.values())
                if total_weight > 0:
                    for model_name in ensemble._model_weights:
                        ensemble._model_weights[model_name] /= total_weight
            else:
                self.logger.warning(f"Cannot remove underperformers: would leave only {remaining_models} models")
            
        except Exception as e:
            self.logger.error(f"Optimization application failed: {e}")
    
    def _update_performance_tracking(self, ensemble_name: str, performance_metrics: Dict[str, List[float]]) -> None:
        """Update performance tracking with new metrics."""
        try:
            if ensemble_name not in self.performance_history:
                self.performance_history[ensemble_name] = []
            
            # Store recent performance metrics for future use
            # This is a simplified version - in practice you might want more sophisticated tracking
            
        except Exception as e:
            self.logger.error(f"Performance tracking update failed: {e}")
    
    def _calculate_prediction_accuracy(self, actual: List[float], predicted: List[float]) -> float:
        """Calculate prediction accuracy score."""
        try:
            if len(actual) != len(predicted) or len(actual) == 0:
                return 0.0
            
            # Calculate MAPE and convert to accuracy score
            mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.where(np.array(actual) != 0, np.array(actual), 1))) * 100
            accuracy = 1.0 / (1.0 + mape / 100.0)
            
            return accuracy
            
        except Exception:
            return 0.0
    
    def _calculate_prediction_diversity(self, individual_predictions: Dict[str, List[float]]) -> float:
        """Calculate diversity score from individual model predictions."""
        try:
            if len(individual_predictions) < 2:
                return 1.0
            
            # Calculate pairwise correlations
            model_names = list(individual_predictions.keys())
            correlations = []
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    pred1 = individual_predictions[model_names[i]]
                    pred2 = individual_predictions[model_names[j]]
                    
                    if len(pred1) == len(pred2) and len(pred1) > 1:
                        corr = np.corrcoef(pred1, pred2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                return 1.0 - avg_correlation
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _calculate_weight_stability(self, ensemble_name: str, current_weights: Dict[str, float]) -> float:
        """Calculate weight stability score."""
        try:
            if ensemble_name not in self.weight_history:
                self.weight_history[ensemble_name] = []
            
            # Add current weights to history
            self.weight_history[ensemble_name].append(current_weights.copy())
            
            # Keep only recent history
            stability_window = self.config.get('stability_window', 10)
            if len(self.weight_history[ensemble_name]) > stability_window:
                self.weight_history[ensemble_name] = self.weight_history[ensemble_name][-stability_window:]
            
            # Calculate stability (low variance = high stability)
            if len(self.weight_history[ensemble_name]) < 2:
                return 1.0
            
            weight_variances = []
            for model_name in current_weights:
                weights_over_time = []
                for historical_weights in self.weight_history[ensemble_name]:
                    if model_name in historical_weights:
                        weights_over_time.append(historical_weights[model_name])
                
                if len(weights_over_time) > 1:
                    variance = np.var(weights_over_time)
                    weight_variances.append(variance)
            
            if weight_variances:
                avg_variance = np.mean(weight_variances)
                stability = 1.0 / (1.0 + avg_variance * 10)  # Scale variance to 0-1 range
                return stability
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def get_ensemble_report(self, ensemble_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive ensemble performance report.
        
        Args:
            ensemble_name: Name of ensemble to report on
            
        Returns:
            Dictionary containing comprehensive ensemble metrics
        """
        try:
            report = {
                'ensemble_name': ensemble_name,
                'timestamp': datetime.now().isoformat(),
                'performance_history': [],
                'diversity_history': [],
                'weight_stability': 0.0,
                'recommendations': []
            }
            
            # Add performance history
            if ensemble_name in self.performance_history:
                recent_performances = self.performance_history[ensemble_name][-10:]
                report['performance_history'] = [perf.to_dict() for perf in recent_performances]
                
                # Calculate average metrics
                if recent_performances:
                    avg_accuracy = np.mean([p.prediction_accuracy for p in recent_performances])
                    avg_diversity = np.mean([p.model_diversity for p in recent_performances])
                    avg_stability = np.mean([p.weight_stability for p in recent_performances])
                    
                    report.update({
                        'average_accuracy': avg_accuracy,
                        'average_diversity': avg_diversity,
                        'average_stability': avg_stability
                    })
            
            # Add diversity history
            if ensemble_name in self.diversity_history:
                recent_diversity = self.diversity_history[ensemble_name][-5:]
                report['diversity_history'] = [div.to_dict() for div in recent_diversity]
            
            # Add weight stability
            if ensemble_name in self.weight_history:
                current_weights = self.weight_history[ensemble_name][-1] if self.weight_history[ensemble_name] else {}
                report['weight_stability'] = self._calculate_weight_stability(ensemble_name, current_weights)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {'error': str(e), 'ensemble_name': ensemble_name}
    
    def save_performance_data(self, filepath: str) -> None:
        """Save performance data to file for persistence."""
        try:
            data = {
                'performance_history': {
                    name: [perf.to_dict() for perf in perfs] 
                    for name, perfs in self.performance_history.items()
                },
                'weight_history': self.weight_history,
                'diversity_history': {
                    name: [div.to_dict() for div in divs]
                    for name, divs in self.diversity_history.items()
                },
                'last_optimization': {
                    name: dt.isoformat() for name, dt in self.last_optimization.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Performance data saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance data: {e}")
    
    def load_performance_data(self, filepath: str) -> None:
        """Load performance data from file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct performance history
                self.performance_history = {}
                for name, perfs in data.get('performance_history', {}).items():
                    self.performance_history[name] = []
                    for perf_dict in perfs:
                        perf_dict['timestamp'] = datetime.fromisoformat(perf_dict['timestamp'])
                        perf = EnsemblePerformance(**perf_dict)
                        self.performance_history[name].append(perf)
                
                # Load other data
                self.weight_history = data.get('weight_history', {})
                
                # Reconstruct diversity history
                self.diversity_history = {}
                for name, divs in data.get('diversity_history', {}).items():
                    self.diversity_history[name] = []
                    for div_dict in divs:
                        div = DiversityMetrics(**div_dict)
                        self.diversity_history[name].append(div)
                
                # Load last optimization times
                self.last_optimization = {}
                for name, dt_str in data.get('last_optimization', {}).items():
                    self.last_optimization[name] = datetime.fromisoformat(dt_str)
                
                self.logger.info(f"Performance data loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load performance data: {e}")