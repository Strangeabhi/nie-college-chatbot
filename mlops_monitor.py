"""
MLOps Monitoring and Management System for NIE Chatbot
Handles performance tracking, auto-retraining, and analytics
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# MLOps imports
import mlflow
import mlflow.sklearn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotMLOpsMonitor:
    def __init__(self, chatbot_instance=None):
        """Initialize MLOps monitoring system"""
        self.chatbot = chatbot_instance
        self.console = Console()
        
        # Metrics tracking
        self.metrics_history = []
        self.performance_thresholds = {
            'accuracy': 0.85,
            'response_time': 2.0,
            'confidence_score': 0.7,
            'error_rate': 0.05
        }
        
        # Auto-retraining settings
        self.auto_retrain_enabled = True
        self.retrain_threshold = 0.1  # Retrain if performance drops by 10%
        self.last_retrain_time = datetime.now()
        self.retrain_interval = timedelta(hours=24)  # Minimum 24h between retrains
        
        # Data collection
        self.interaction_log = []
        self.feedback_log = []
        
        # Initialize MLflow
        self.setup_mlflow()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("MLOps Monitor initialized")

    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("nie_chatbot_monitoring")
            logger.info("MLflow tracking setup completed")
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")

    def log_interaction(self, user_query: str, bot_response: str, 
                       confidence: float, response_time: float, 
                       user_id: str = "anonymous"):
        """Log user interaction for analysis"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'query': user_query,
            'response': bot_response,
            'confidence': confidence,
            'response_time': response_time,
            'query_length': len(user_query),
            'response_length': len(bot_response)
        }
        
        self.interaction_log.append(interaction)
        
        # Keep only last 1000 interactions
        if len(self.interaction_log) > 1000:
            self.interaction_log = self.interaction_log[-1000:]
        
        # Log to MLflow
        try:
            with mlflow.start_run():
                mlflow.log_param("user_query", user_query[:100])  # Truncate for MLflow
                mlflow.log_metric("confidence_score", confidence)
                mlflow.log_metric("response_time", response_time)
                mlflow.log_metric("query_length", len(user_query))
                mlflow.log_metric("response_length", len(bot_response))
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def log_feedback(self, user_query: str, bot_response: str, 
                    feedback_score: float, user_id: str = "anonymous"):
        """Log user feedback for model improvement"""
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'query': user_query,
            'response': bot_response,
            'feedback_score': feedback_score  # 1-5 scale or 0/1 for good/bad
        }
        
        self.feedback_log.append(feedback)
        
        # Keep only last 500 feedback entries
        if len(self.feedback_log) > 500:
            self.feedback_log = self.feedback_log[-500:]

    def get_performance_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if not self.interaction_log:
            return {
                'total_interactions': 0,
                'avg_confidence': 0,
                'avg_response_time': 0,
                'error_rate': 0,
                'success_rate': 0
            }
        
        recent_interactions = self.interaction_log[-100:]  # Last 100 interactions
        
        metrics = {
            'total_interactions': len(self.interaction_log),
            'recent_interactions': len(recent_interactions),
            'avg_confidence': np.mean([i['confidence'] for i in recent_interactions]),
            'avg_response_time': np.mean([i['response_time'] for i in recent_interactions]),
            'error_rate': len([i for i in recent_interactions if i['confidence'] < 0.5]) / len(recent_interactions),
            'success_rate': len([i for i in recent_interactions if i['confidence'] >= 0.7]) / len(recent_interactions),
            'avg_query_length': np.mean([i['query_length'] for i in recent_interactions]),
            'avg_response_length': np.mean([i['response_length'] for i in recent_interactions])
        }
        
        # Add feedback metrics if available
        if self.feedback_log:
            recent_feedback = self.feedback_log[-50:]
            metrics['avg_feedback_score'] = np.mean([f['feedback_score'] for f in recent_feedback])
            metrics['feedback_count'] = len(self.feedback_log)
        
        return metrics

    def check_retrain_conditions(self) -> bool:
        """Check if model should be retrained"""
        if not self.auto_retrain_enabled:
            return False
        
        # Check time interval
        if datetime.now() - self.last_retrain_time < self.retrain_interval:
            return False
        
        # Check performance degradation
        metrics = self.get_performance_metrics()
        
        if metrics['avg_confidence'] < self.performance_thresholds['confidence_score']:
            logger.warning(f"Low confidence detected: {metrics['avg_confidence']:.2f}")
            return True
        
        if metrics['error_rate'] > self.performance_thresholds['error_rate']:
            logger.warning(f"High error rate detected: {metrics['error_rate']:.2f}")
            return True
        
        return False

    def trigger_retrain(self) -> bool:
        """Trigger model retraining"""
        try:
            logger.info("Triggering model retraining...")
            
            if self.chatbot and hasattr(self.chatbot, 'retrain_model'):
                success = self.chatbot.retrain_model()
                if success:
                    self.last_retrain_time = datetime.now()
                    logger.info("Model retraining completed successfully")
                    return True
                else:
                    logger.error("Model retraining failed")
                    return False
            else:
                logger.warning("Chatbot instance not available for retraining")
                return False
                
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return False

    def get_analytics_report(self) -> Dict:
        """Generate comprehensive analytics report"""
        metrics = self.get_performance_metrics()
        
        # Top queries analysis
        if self.interaction_log:
            query_counts = {}
            for interaction in self.interaction_log[-100:]:
                query = interaction['query'].lower()[:50]  # First 50 chars
                query_counts[query] = query_counts.get(query, 0) + 1
            
            top_queries = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            top_queries = []
        
        # Category analysis (if chatbot available)
        category_stats = {}
        if self.chatbot and hasattr(self.chatbot, 'categories'):
            category_counts = {}
            for interaction in self.interaction_log[-100:]:
                # This would need to be implemented to map queries to categories
                pass
        
        return {
            'performance_metrics': metrics,
            'top_queries': top_queries,
            'category_stats': category_stats,
            'feedback_summary': self._get_feedback_summary(),
            'recommendations': self._get_recommendations(metrics)
        }

    def _get_feedback_summary(self) -> Dict:
        """Get feedback summary"""
        if not self.feedback_log:
            return {'total_feedback': 0, 'avg_score': 0}
        
        scores = [f['feedback_score'] for f in self.feedback_log]
        return {
            'total_feedback': len(self.feedback_log),
            'avg_score': np.mean(scores),
            'positive_feedback': len([s for s in scores if s >= 3]),
            'negative_feedback': len([s for s in scores if s < 3])
        }

    def _get_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        if metrics['avg_confidence'] < 0.7:
            recommendations.append("Consider adding more FAQ variations to improve response accuracy")
        
        if metrics['avg_response_time'] > 2.0:
            recommendations.append("Response time is high - consider optimizing embeddings or using caching")
        
        if metrics['error_rate'] > 0.1:
            recommendations.append("High error rate detected - review failed queries and improve FAQ coverage")
        
        if not self.feedback_log:
            recommendations.append("Implement user feedback collection to improve model performance")
        
        return recommendations

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Check for retraining conditions
                if self.check_retrain_conditions():
                    self.trigger_retrain()
                
                # Log system metrics
                system_metrics = {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store metrics
                self.metrics_history.append(system_metrics)
                
                # Keep only last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)

    def display_dashboard(self):
        """Display real-time monitoring dashboard"""
        layout = Layout()
        
        layout.split_column(
            Layout(Panel("NIE Chatbot MLOps Dashboard", style="bold blue"), size=3),
            Layout(name="main"),
            Layout(Panel("System Status", style="green"), size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        with Live(layout, refresh_per_second=1) as live:
            while True:
                try:
                    # Performance metrics
                    metrics = self.get_performance_metrics()
                    
                    # Performance table
                    perf_table = Table(title="Performance Metrics")
                    perf_table.add_column("Metric", style="cyan")
                    perf_table.add_column("Value", style="magenta")
                    perf_table.add_column("Status", style="green")
                    
                    perf_table.add_row("Total Interactions", str(metrics['total_interactions']), "✓")
                    perf_table.add_row("Avg Confidence", f"{metrics['avg_confidence']:.2f}", 
                                     "✓" if metrics['avg_confidence'] > 0.7 else "⚠")
                    perf_table.add_row("Avg Response Time", f"{metrics['avg_response_time']:.2f}s", 
                                     "✓" if metrics['avg_response_time'] < 2.0 else "⚠")
                    perf_table.add_row("Success Rate", f"{metrics['success_rate']:.2%}", 
                                     "✓" if metrics['success_rate'] > 0.8 else "⚠")
                    
                    # Top queries table
                    analytics = self.get_analytics_report()
                    queries_table = Table(title="Top Queries")
                    queries_table.add_column("Query", style="cyan")
                    queries_table.add_column("Count", style="magenta")
                    
                    for query, count in analytics['top_queries'][:5]:
                        queries_table.add_row(query[:30] + "...", str(count))
                    
                    layout["left"].update(Panel(perf_table, title="Performance"))
                    layout["right"].update(Panel(queries_table, title="Analytics"))
                    
                    # System status
                    if self.metrics_history:
                        latest_metrics = self.metrics_history[-1]
                        status = f"CPU: {latest_metrics['cpu_usage']:.1f}% | Memory: {latest_metrics['memory_usage']:.1f}% | Disk: {latest_metrics['disk_usage']:.1f}%"
                        layout["System Status"].update(Panel(status, style="green"))
                    
                    time.sleep(1)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Dashboard error: {e}")
                    time.sleep(1)

    def export_analytics(self, filename: str = None) -> str:
        """Export analytics data to JSON file"""
        if filename is None:
            filename = f"analytics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        analytics_data = {
            'export_timestamp': datetime.now().isoformat(),
            'analytics_report': self.get_analytics_report(),
            'interaction_log': self.interaction_log[-100:],  # Last 100 interactions
            'feedback_log': self.feedback_log[-50:],  # Last 50 feedback entries
            'system_metrics': self.metrics_history[-50:]  # Last 50 system metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analytics_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analytics exported to {filename}")
        return filename

    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("MLOps monitoring stopped")


# Global monitor instance
monitor = None

def initialize_monitor(chatbot_instance=None):
    """Initialize the MLOps monitor"""
    global monitor
    if monitor is None:
        monitor = ChatbotMLOpsMonitor(chatbot_instance)
    return monitor

def get_monitor():
    """Get the monitor instance"""
    return monitor

if __name__ == "__main__":
    # Test the MLOps monitor
    monitor = initialize_monitor()
    
    print("MLOps Monitor initialized!")
    print("Starting dashboard... (Press Ctrl+C to stop)")
    
    try:
        monitor.display_dashboard()
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        monitor.stop_monitoring()
