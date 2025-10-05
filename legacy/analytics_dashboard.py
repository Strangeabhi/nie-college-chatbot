"""
Visual Analytics Dashboard for NIE Chatbot
Provides insights into chatbot performance, user queries, and system metrics
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    def __init__(self, data_dir: str = "analytics_data"):
        """Initialize the analytics dashboard"""
        self.data_dir = data_dir
        self.console = Console()
        self.ensure_data_dir()
        
        # Load existing data
        self.interactions = self.load_interactions()
        self.feedback_data = self.load_feedback()
        self.performance_metrics = self.load_performance_metrics()
        
        logger.info("Analytics Dashboard initialized")

    def ensure_data_dir(self):
        """Ensure analytics data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_interactions(self) -> List[Dict]:
        """Load interaction data"""
        try:
            interactions_file = os.path.join(self.data_dir, "interactions.json")
            if os.path.exists(interactions_file):
                with open(interactions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Error loading interactions: {e}")
            return []

    def load_feedback(self) -> List[Dict]:
        """Load feedback data"""
        try:
            feedback_file = os.path.join(self.data_dir, "feedback.json")
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.warning(f"Error loading feedback: {e}")
            return []

    def load_performance_metrics(self) -> Dict:
        """Load performance metrics"""
        try:
            metrics_file = os.path.join(self.data_dir, "performance_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error loading performance metrics: {e}")
            return {}

    def add_interaction(self, interaction_data: Dict):
        """Add new interaction data"""
        self.interactions.append(interaction_data)
        self.save_interactions()

    def add_feedback(self, feedback_data: Dict):
        """Add new feedback data"""
        self.feedback_data.append(feedback_data)
        self.save_feedback()

    def save_interactions(self):
        """Save interactions to file"""
        try:
            interactions_file = os.path.join(self.data_dir, "interactions.json")
            with open(interactions_file, 'w', encoding='utf-8') as f:
                json.dump(self.interactions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving interactions: {e}")

    def save_feedback(self):
        """Save feedback to file"""
        try:
            feedback_file = os.path.join(self.data_dir, "feedback.json")
            with open(feedback_file, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")

    def get_query_analytics(self) -> Dict:
        """Analyze user queries"""
        if not self.interactions:
            return {}
        
        # Extract queries
        queries = [interaction.get('query', '') for interaction in self.interactions]
        
        # Most common queries
        query_counts = Counter(queries)
        top_queries = query_counts.most_common(10)
        
        # Query length analysis
        query_lengths = [len(q.split()) for q in queries]
        avg_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0
        
        # Query categories (simple keyword-based)
        categories = defaultdict(int)
        for query in queries:
            query_lower = query.lower()
            if any(word in query_lower for word in ['cutoff', 'rank', 'admission']):
                categories['Admissions'] += 1
            elif any(word in query_lower for word in ['placement', 'package', 'job']):
                categories['Placements'] += 1
            elif any(word in query_lower for word in ['hostel', 'room', 'accommodation']):
                categories['Hostels'] += 1
            elif any(word in query_lower for word in ['fee', 'cost', 'payment']):
                categories['Fees'] += 1
            else:
                categories['Other'] += 1
        
        return {
            'total_queries': len(queries),
            'unique_queries': len(set(queries)),
            'top_queries': top_queries,
            'avg_query_length': avg_query_length,
            'categories': dict(categories)
        }

    def get_confidence_analytics(self) -> Dict:
        """Analyze confidence scores"""
        if not self.interactions:
            return {}
        
        confidences = [interaction.get('confidence', 0) for interaction in self.interactions]
        
        if not confidences:
            return {}
        
        # Confidence distribution
        high_confidence = len([c for c in confidences if c >= 0.8])
        medium_confidence = len([c for c in confidences if 0.5 <= c < 0.8])
        low_confidence = len([c for c in confidences if c < 0.5])
        
        return {
            'avg_confidence': sum(confidences) / len(confidences),
            'high_confidence_count': high_confidence,
            'medium_confidence_count': medium_confidence,
            'low_confidence_count': low_confidence,
            'confidence_distribution': {
                'High (≥0.8)': high_confidence,
                'Medium (0.5-0.8)': medium_confidence,
                'Low (<0.5)': low_confidence
            }
        }

    def get_response_time_analytics(self) -> Dict:
        """Analyze response times"""
        if not self.interactions:
            return {}
        
        response_times = [interaction.get('response_time', 0) for interaction in self.interactions]
        response_times = [rt for rt in response_times if rt > 0]  # Filter out invalid times
        
        if not response_times:
            return {}
        
        return {
            'avg_response_time': sum(response_times) / len(response_times),
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'fast_responses': len([rt for rt in response_times if rt < 1.0]),
            'slow_responses': len([rt for rt in response_times if rt > 3.0])
        }

    def get_feedback_analytics(self) -> Dict:
        """Analyze user feedback"""
        if not self.feedback_data:
            return {'total_feedback': 0, 'avg_rating': 0}
        
        ratings = [feedback.get('score', 0) for feedback in self.feedback_data]
        
        return {
            'total_feedback': len(self.feedback_data),
            'avg_rating': sum(ratings) / len(ratings) if ratings else 0,
            'positive_feedback': len([r for r in ratings if r >= 4]),
            'negative_feedback': len([r for r in ratings if r <= 2]),
            'rating_distribution': dict(Counter(ratings))
        }

    def generate_report(self) -> Dict:
        """Generate comprehensive analytics report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'query_analytics': self.get_query_analytics(),
            'confidence_analytics': self.get_confidence_analytics(),
            'response_time_analytics': self.get_response_time_analytics(),
            'feedback_analytics': self.get_feedback_analytics(),
            'system_health': self.get_system_health()
        }

    def get_system_health(self) -> Dict:
        """Get system health metrics"""
        recent_interactions = [
            i for i in self.interactions 
            if datetime.fromisoformat(i.get('timestamp', '2020-01-01')) > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'interactions_last_24h': len(recent_interactions),
            'total_interactions': len(self.interactions),
            'data_freshness': 'Good' if recent_interactions else 'No recent data',
            'system_status': 'Healthy' if len(self.interactions) > 0 else 'No data'
        }

    def display_dashboard(self):
        """Display real-time analytics dashboard"""
        layout = Layout()
        
        layout.split_column(
            Layout(Panel("NIE Chatbot Analytics Dashboard", style="bold blue"), size=3),
            Layout(name="main"),
            Layout(Panel("Last Updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="green"), size=1)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        with Live(layout, refresh_per_second=1) as live:
            while True:
                try:
                    report = self.generate_report()
                    
                    # Left panel - Query Analytics
                    query_table = Table(title="Query Analytics")
                    query_table.add_column("Metric", style="cyan")
                    query_table.add_column("Value", style="magenta")
                    
                    query_analytics = report.get('query_analytics', {})
                    query_table.add_row("Total Queries", str(query_analytics.get('total_queries', 0)))
                    query_table.add_row("Unique Queries", str(query_analytics.get('unique_queries', 0)))
                    query_table.add_row("Avg Query Length", f"{query_analytics.get('avg_query_length', 0):.1f} words")
                    
                    # Top queries
                    top_queries = query_analytics.get('top_queries', [])[:3]
                    for query, count in top_queries:
                        query_table.add_row(f"Top Query", f"{query[:30]}... ({count})")
                    
                    # Right panel - Performance Metrics
                    perf_table = Table(title="Performance Metrics")
                    perf_table.add_column("Metric", style="cyan")
                    perf_table.add_column("Value", style="magenta")
                    
                    conf_analytics = report.get('confidence_analytics', {})
                    perf_table.add_row("Avg Confidence", f"{conf_analytics.get('avg_confidence', 0):.2f}")
                    perf_table.add_row("High Confidence", str(conf_analytics.get('high_confidence_count', 0)))
                    perf_table.add_row("Low Confidence", str(conf_analytics.get('low_confidence_count', 0)))
                    
                    rt_analytics = report.get('response_time_analytics', {})
                    perf_table.add_row("Avg Response Time", f"{rt_analytics.get('avg_response_time', 0):.2f}s")
                    perf_table.add_row("Fast Responses", str(rt_analytics.get('fast_responses', 0)))
                    
                    feedback_analytics = report.get('feedback_analytics', {})
                    perf_table.add_row("Avg Rating", f"{feedback_analytics.get('avg_rating', 0):.1f}/5")
                    perf_table.add_row("Total Feedback", str(feedback_analytics.get('total_feedback', 0)))
                    
                    layout["left"].update(Panel(query_table, title="Query Analytics"))
                    layout["right"].update(Panel(perf_table, title="Performance Metrics"))
                    
                    time.sleep(5)  # Update every 5 seconds
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logger.error(f"Dashboard error: {e}")
                    time.sleep(1)

    def export_report(self, filename: str = None) -> str:
        """Export analytics report to JSON file"""
        if filename is None:
            filename = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analytics report exported to {filename}")
        return filename

    def create_visualizations(self, output_dir: str = "charts"):
        """Create visual analytics charts"""
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            report = self.generate_report()
            
            # Query categories pie chart
            categories = report.get('query_analytics', {}).get('categories', {})
            if categories:
                plt.figure(figsize=(10, 6))
                plt.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
                plt.title('Query Categories Distribution')
                plt.savefig(os.path.join(output_dir, 'query_categories.png'))
                plt.close()
            
            # Confidence distribution
            conf_dist = report.get('confidence_analytics', {}).get('confidence_distribution', {})
            if conf_dist:
                plt.figure(figsize=(10, 6))
                plt.bar(conf_dist.keys(), conf_dist.values())
                plt.title('Confidence Score Distribution')
                plt.xlabel('Confidence Level')
                plt.ylabel('Number of Responses')
                plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
                plt.close()
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")

# Global dashboard instance
dashboard = None

def get_dashboard():
    """Get global dashboard instance"""
    global dashboard
    if dashboard is None:
        dashboard = AnalyticsDashboard()
    return dashboard

if __name__ == "__main__":
    # Test the analytics dashboard
    dashboard = get_dashboard()
    
    # Add some sample data
    sample_interactions = [
        {
            'timestamp': datetime.now().isoformat(),
            'query': 'What are CSE cutoffs?',
            'confidence': 0.9,
            'response_time': 1.2
        },
        {
            'timestamp': datetime.now().isoformat(),
            'query': 'Placement statistics',
            'confidence': 0.8,
            'response_time': 0.8
        }
    ]
    
    for interaction in sample_interactions:
        dashboard.add_interaction(interaction)
    
    # Generate and display report
    report = dashboard.generate_report()
    print(json.dumps(report, indent=2))
    
    # Display dashboard
    print("\nStarting dashboard... (Press Ctrl+C to stop)")
    try:
        dashboard.display_dashboard()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
