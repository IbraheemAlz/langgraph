"""
Chart Generator for the Multi-Agent AI Hiring System.
Pure visualization from JSON results - creates all charts immediately.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

class HiringSystemChartGenerator:
    """Complete chart generation for the hiring system."""
    
    def __init__(self):
        """Initialize the chart generator with seaborn styling."""
        sns.set_style("whitegrid")
        plt.style.use("seaborn-v0_8")
    
    def create_evaluation_charts(self, results_file: str = "results/json/batch_results.json", 
                                output_file: str = "results/images/evaluation_results.png"):
        """Create the enhanced 6-panel evaluation charts from JSON results."""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            if not results:
                logger.warning("No results found in file")
                return
            
            # Ensure output directory exists
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(results)
            
            # Create 4-panel layout (2x2) - returning to original format
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.patch.set_facecolor('white')
            fig.suptitle('Multi-Agent AI Hiring System - Comprehensive Evaluation Report', 
                        fontsize=18, fontweight='bold', y=0.95)
            
            # Add subtle dividing lines between charts
            fig.add_artist(plt.Line2D([0.5, 0.5], [0.02, 0.93], 
                                    transform=fig.transFigure, color='lightgray', linewidth=1))
            fig.add_artist(plt.Line2D([0, 1], [0.5, 0.5], 
                                    transform=fig.transFigure, color='lightgray', linewidth=1))
            
            # 1. Decision Distribution (Top Left)
            decision_counts = df['final_decision'].value_counts()
            total_decisions = len(df)
            colors1 = ['#5B9BD5', '#70AD47']  # Blue for select, Green for reject
            wedges, texts, autotexts = axes[0, 0].pie(decision_counts.values, labels=decision_counts.index, 
                                                     autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_decisions)})', 
                                                     colors=colors1, startangle=90)
            axes[0, 0].set_title('Decision Distribution', fontweight='bold', fontsize=14)
            # Add equation
            axes[0, 0].text(0.5, -1.3, 'Formula: (Decision Count / Total Candidates) × 100%', 
                           ha='center', va='center', transform=axes[0, 0].transAxes,
                           fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))
            
            # 2. Bias Classification Distribution (Top Right)
            bias_counts = df['bias_classification'].value_counts()
            colors2 = ['#70AD47', '#E15759']  # Green for unbiased, Red for biased
            wedges2, texts2, autotexts2 = axes[0, 1].pie(bias_counts.values, labels=bias_counts.index, 
                                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total_decisions)})', 
                                                         colors=colors2, startangle=90)
            axes[0, 1].set_title('Bias Classification Distribution', fontweight='bold', fontsize=14)
            # Add equation
            axes[0, 1].text(0.5, -1.3, 'Formula: (Bias Classification Count / Total Candidates) × 100%', 
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
            
            # 3. Re-evaluation Frequency (Bottom Left)
            # Transform re_evaluation_count to "attempts until acceptance"
            # re_evaluation_count 0 = accepted on 1st attempt
            # re_evaluation_count 1 = accepted on 2nd attempt  
            # re_evaluation_count 2 = accepted on 3rd attempt
            df['attempts_until_acceptance'] = df['re_evaluation_count'] + 1
            
            attempt_counts = df['attempts_until_acceptance'].value_counts().sort_index()
            
            # Create labels for better understanding
            attempt_labels = []
            for attempt in attempt_counts.index:
                if attempt == 1:
                    attempt_labels.append(f'{attempt}st Attempt')
                elif attempt == 2:
                    attempt_labels.append(f'{attempt}nd Attempt')
                elif attempt == 3:
                    attempt_labels.append(f'{attempt}rd Attempt')
                else:
                    attempt_labels.append(f'{attempt}th Attempt')
            
            bars = axes[1, 0].bar(range(len(attempt_counts)), attempt_counts.values, 
                                 color='#5B9BD5', alpha=0.8, width=0.6)
            axes[1, 0].set_title('Re-evaluation Frequency', fontweight='bold', fontsize=14)
            axes[1, 0].set_xlabel('Decision Accepted After', fontweight='bold')
            axes[1, 0].set_ylabel('Count', fontweight='bold')
            axes[1, 0].set_xticks(range(len(attempt_counts)))
            axes[1, 0].set_xticklabels(attempt_labels, rotation=45, ha='right')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_facecolor('#f8f9fa')
            
            # Add count and percentage labels on top of bars
            for bar, count in zip(bars, attempt_counts.values):
                height = bar.get_height()
                percentage = count / total_decisions * 100
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + max(attempt_counts.values) * 0.01,
                               f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Add mathematical equation with actual numbers - improved formatting
            total_1st = attempt_counts.get(1, 0)
            total_2nd = attempt_counts.get(2, 0) 
            total_3rd = attempt_counts.get(3, 0)
            
            pct_1st = (total_1st / total_decisions * 100) if total_1st > 0 else 0
            pct_2nd = (total_2nd / total_decisions * 100) if total_2nd > 0 else 0
            pct_3rd = (total_3rd / total_decisions * 100) if total_3rd > 0 else 0
            
            formula_text = f"Distribution: 1st={total_1st}({pct_1st:.1f}%) | 2nd={total_2nd}({pct_2nd:.1f}%) | 3rd={total_3rd}({pct_3rd:.1f}%)"
            axes[1, 0].text(0.5, -0.15, formula_text, 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=8, style='italic', bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
            
            # 4. System Accuracy (Bottom Right)
            if 'ground_truth_decision' in df.columns and 'ground_truth_bias' in df.columns:
                # Calculate decision accuracy
                correct_decisions = (df['final_decision'] == df['ground_truth_decision']).sum()
                decision_accuracy = correct_decisions / total_decisions
                
                # Calculate bias classification accuracy
                correct_bias = (df['bias_classification'] == df['ground_truth_bias']).sum()
                bias_accuracy = correct_bias / total_decisions
                
                # Calculate correction score - keeping the same equation
                correction_cases = df[
                    (df['ground_truth_bias'] == 'biased') &  # Original was biased
                    (df['bias_classification'] == 'unbiased') &  # System corrected to unbiased
                    (df['ground_truth_decision'] != df['final_decision'])  # Decision actually changed
                ]
                total_biased_cases = (df['ground_truth_bias'] == 'biased').sum()
                correction_score = len(correction_cases) / total_biased_cases if total_biased_cases > 0 else 0
                
                # Create accuracy bars with correction score as yellow
                metrics = ['Decision\nAccuracy', 'Bias Detection\nAccuracy', 'Correction\nScore']
                values = [decision_accuracy, bias_accuracy, correction_score]
                colors = ['#70AD47', '#E15759', '#FFD700']  # Green, Red, Yellow
                
                bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.8, width=0.6)
                axes[1, 1].set_title('System Accuracy', fontweight='bold', fontsize=14)
                axes[1, 1].set_ylabel('Accuracy', fontweight='bold')
                axes[1, 1].set_ylim(0, 1.0)
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_facecolor('#f8f9fa')
                
                # Add percentage and count labels on bars
                counts = [correct_decisions, correct_bias, len(correction_cases)]
                denominators = [total_decisions, total_decisions, total_biased_cases if total_biased_cases > 0 else 1]
                for bar, value, count, denom in zip(bars, values, counts, denominators):
                    height = bar.get_height()
                    if bar == bars[2]:  # Correction score bar
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                       f'{value:.1%}\n({count}/{denom})', ha='center', va='bottom', 
                                       fontweight='bold', fontsize=10)
                    else:
                        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                       f'{value:.1%}\n({count}/{total_decisions})', ha='center', va='bottom', 
                                       fontweight='bold', fontsize=10)
                
                # Add mathematical equation with actual numbers - improved formatting
                correction_formula = f'Correction Score = {len(correction_cases)} ÷ {total_biased_cases} = {correction_score:.3f}'
                axes[1, 1].text(0.5, -0.15, correction_formula, 
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=9, style='italic', bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcoral', alpha=0.8))
            else:
                # Fallback if no ground truth
                axes[1, 1].text(0.5, 0.5, 'Ground Truth\nNot Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=16, fontweight='bold')
                axes[1, 1].set_title('System Accuracy', fontweight='bold', fontsize=14)
                axes[1, 1].set_facecolor('#f8f9fa')
            
            # Adjust spacing between subplots - increased bottom margin for formulas
            plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.88, wspace=0.25, hspace=0.45)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"📈 Enhanced evaluation charts saved to: {output_file}")
            plt.close()  # Close instead of show to prevent blocking
            
        except FileNotFoundError:
            logger.error(f"Results file {results_file} not found")
            raise
        except Exception as e:
            logger.error(f"Error creating evaluation charts: {e}")
            raise
    
    def create_workflow_diagram(self, save_path: str = "results/images/workflow_diagram.png"):
        """Create an enhanced visual diagram of the LangGraph workflow."""
        # Ensure output directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Larger figure size for better readability
        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        fig.patch.set_facecolor('white')
        
        # Define positions for nodes with better spacing
        positions = {
            "START": (1, 5),
            "job_matcher": (4, 5),
            "bias_classifier": (7, 5),
            "should_continue": (10, 5),
            "re_evaluate": (7, 8),
            "finalize": (13, 5),
            "END": (16, 5)
        }
        
        # Enhanced node styling
        node_styles = {
            "START": {"color": "#4CAF50", "marker": "o", "size": 2500, "label": "START\n(Input Data)"},
            "job_matcher": {"color": "#2196F3", "marker": "s", "size": 3500, "label": "Job Matcher\nAgent"},
            "bias_classifier": {"color": "#FF9800", "marker": "s", "size": 3500, "label": "Bias Classifier\nAgent"},
            "should_continue": {"color": "#FFC107", "marker": "D", "size": 3000, "label": "Should\nContinue?"},
            "re_evaluate": {"color": "#E91E63", "marker": "s", "size": 3000, "label": "Re-evaluate\nNode"},
            "finalize": {"color": "#9C27B0", "marker": "s", "size": 3000, "label": "Finalize\nDecision"},
            "END": {"color": "#4CAF50", "marker": "o", "size": 2500, "label": "END\n(Final Output)"}
        }
        
        # Draw enhanced nodes
        for node, (x, y) in positions.items():
            style = node_styles[node]
            ax.scatter(x, y, s=style["size"], c=style["color"], marker=style["marker"], 
                      edgecolors='black', linewidth=2, alpha=0.9, zorder=3)
            ax.text(x, y-0.7, style["label"], ha='center', va='top', fontsize=10, 
                   fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Enhanced arrows with labels
        arrows = [
            {"start": "START", "end": "job_matcher", "label": "Initial\nEvaluation", "color": "blue"},
            {"start": "job_matcher", "end": "bias_classifier", "label": "Decision +\nReasoning", "color": "blue"},
            {"start": "bias_classifier", "end": "should_continue", "label": "Bias\nClassification", "color": "blue"},
            {"start": "should_continue", "end": "finalize", "label": "No Bias\nDetected", "color": "green"},
            {"start": "should_continue", "end": "re_evaluate", "label": "Bias Detected &\n< Max Attempts", "color": "red"},
            {"start": "re_evaluate", "end": "job_matcher", "label": "Feedback Loop\n(with bias context)", "color": "red"},
            {"start": "finalize", "end": "END", "label": "Process\nComplete", "color": "green"}
        ]
        
        for arrow in arrows:
            x1, y1 = positions[arrow["start"]]
            x2, y2 = positions[arrow["end"]]
            
            # Calculate arrow positioning to avoid node overlap
            if arrow["start"] == "should_continue" and arrow["end"] == "re_evaluate":
                # Upward arrow
                ax.annotate('', xy=(x2, y2-0.4), xytext=(x1-0.2, y1+0.4),
                           arrowprops=dict(arrowstyle='->', lw=2.5, color=arrow["color"]))
                ax.text(x1-0.5, y1+1.5, arrow["label"], ha='center', va='center', fontsize=10, 
                       color=arrow["color"], fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
            elif arrow["start"] == "re_evaluate" and arrow["end"] == "job_matcher":
                # Curved feedback arrow
                ax.annotate('', xy=(x2-0.1, y2+0.4), xytext=(x1-0.1, y1-0.4),
                           arrowprops=dict(arrowstyle='->', lw=2.5, color=arrow["color"],
                                         connectionstyle="arc3,rad=-0.3"))
                ax.text(5, 6.8, arrow["label"], ha='center', va='center', fontsize=10, 
                       color=arrow["color"], fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
            else:
                # Regular horizontal arrows
                ax.annotate('', xy=(x2-0.5, y2), xytext=(x1+0.5, y1),
                           arrowprops=dict(arrowstyle='->', lw=2.5, color=arrow["color"]))
                mid_x = (x1 + x2) / 2
                ax.text(mid_x, y1+0.7, arrow["label"], ha='center', va='center', fontsize=10, 
                       color=arrow["color"], fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
        
        # Add system information box
        info_text = """🤖 LangGraph Multi-Agent System Features:
• State Management: InMemorySaver with checkpointing
• Max Re-evaluations: 2 attempts per candidate
• Model: Google Gemma 3 (27B-IT) via API
• API Management: Single API key configuration
• Memory: Persistent evaluation insights tracking"""
        
        ax.text(0.01, 0.99, info_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='lightblue', alpha=0.8), fontfamily='monospace')
        
        # Add conditional logic explanation
        logic_text = """🔀 Conditional Logic:
should_continue() checks:
1. Bias classification result
2. Current re-evaluation count
3. Max attempts limit (2)
4. Returns: "re_evaluate" or "finalize" """
        
        ax.text(0.01, 0.01, logic_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='lightyellow', alpha=0.8), fontfamily='monospace')
        
        ax.set_xlim(-0.5, 17.5)
        ax.set_ylim(3, 9.5)
        ax.set_title("Multi-Agent AI Hiring System - Enhanced LangGraph Workflow", 
                    fontsize=20, fontweight='bold', pad=25)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"🔄 Enhanced workflow diagram saved to: {save_path}")
        plt.close()
    
    def create_system_architecture_diagram(self, save_path: str = "results/images/system_architecture.png"):
        """Create a comprehensive system architecture diagram reflecting current implementation."""
        # Ensure output directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Significantly larger figure size for better spacing
        fig, ax = plt.subplots(1, 1, figsize=(20, 16))
        fig.patch.set_facecolor('white')
        
        # Define components and their positions with much better spacing
        components = {
            # Input layer (Top) - spread wider
            "Resume": (2, 13),
            "Job Description": (5, 13),
            "Interview Transcript": (8, 13),
            "Role": (11, 13),
            
            # API Management Layer - better spacing
            "API Key Manager": (1, 11),
            "Rate Limiter": (7, 11),
            
            # Agent layer - more separation
            "Job Matching Agent": (3, 9),
            "Bias Classification Agent": (9, 9),
            
            # LLM layer (Updated to Gemma) - center position
            "Google Gemma 3\n(27B-IT)": (6, 7),
            
            # LangGraph Core - spread horizontally
            "LangGraph StateGraph": (6, 5),
            "InMemorySaver\nCheckpointer": (10, 5),
            "Conditional Routing\n(should_continue)": (2, 5),
            
            # State Management - better horizontal spacing
            "HiringState\nManagement": (6, 3),
            "Evaluation Insights\nTracking": (2, 3),
            "Re-evaluation\nCounter": (10, 3),
            
            # Output layer (Bottom) - spread across width
            "Final Decision": (2, 1),
            "Bias Classification": (5, 1),
            "Audit Trail": (8, 1),
            "Process Metadata": (11, 1)
        }
        
        # Enhanced component styling with larger sizes for bigger chart
        component_styles = {
            # Input components (green)
            "Resume": {"color": "#4CAF50", "size": 2500, "shape": "o"},
            "Job Description": {"color": "#4CAF50", "size": 2500, "shape": "o"},
            "Interview Transcript": {"color": "#4CAF50", "size": 2500, "shape": "o"},
            "Role": {"color": "#4CAF50", "size": 2500, "shape": "o"},
            
            # API Management (orange)
            "API Key Manager": {"color": "#FF9800", "size": 3000, "shape": "s"},
            "Rate Limiter": {"color": "#FF9800", "size": 3000, "shape": "s"},
            
            # Agents (blue)
            "Job Matching Agent": {"color": "#2196F3", "size": 4000, "shape": "s"},
            "Bias Classification Agent": {"color": "#2196F3", "size": 4000, "shape": "s"},
            
            # LLM (red)
            "Google Gemma 3\n(27B-IT)": {"color": "#F44336", "size": 5000, "shape": "o"},
            
            # LangGraph Core (purple)
            "LangGraph StateGraph": {"color": "#9C27B0", "size": 4000, "shape": "s"},
            "InMemorySaver\nCheckpointer": {"color": "#9C27B0", "size": 3500, "shape": "s"},
            "Conditional Routing\n(should_continue)": {"color": "#9C27B0", "size": 3500, "shape": "D"},
            
            # State Management (yellow)
            "HiringState\nManagement": {"color": "#FFEB3B", "size": 3500, "shape": "s"},
            "Evaluation Insights\nTracking": {"color": "#FFEB3B", "size": 3000, "shape": "s"},
            "Re-evaluation\nCounter": {"color": "#FFEB3B", "size": 3000, "shape": "s"},
            
            # Output (gray)
            "Final Decision": {"color": "#9E9E9E", "size": 2500, "shape": "o"},
            "Bias Classification": {"color": "#9E9E9E", "size": 2500, "shape": "o"},
            "Audit Trail": {"color": "#9E9E9E", "size": 2500, "shape": "o"},
            "Process Metadata": {"color": "#9E9E9E", "size": 2500, "shape": "o"}
        }
        
        # Draw enhanced components with better text positioning
        for comp, (x, y) in components.items():
            style = component_styles[comp]
            ax.scatter(x, y, s=style["size"], c=style["color"], marker=style["shape"],
                      edgecolors='black', linewidth=2, alpha=0.9, zorder=3)
            
            # Enhanced text with background and better spacing
            ax.text(x, y-0.8, comp, ha='center', va='top', fontsize=10, 
                   fontweight='bold', wrap=True,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Enhanced connections with different line styles
        connections = [
            # Input to API Management
            {"start": "Resume", "end": "API Key Manager", "style": "--", "color": "gray", "width": 1},
            {"start": "Interview Transcript", "end": "Rate Limiter", "style": "--", "color": "gray", "width": 1},
            
            # Input to agents (main flow)
            {"start": "Resume", "end": "Job Matching Agent", "style": "-", "color": "blue", "width": 2},
            {"start": "Job Description", "end": "Job Matching Agent", "style": "-", "color": "blue", "width": 2},
            {"start": "Interview Transcript", "end": "Job Matching Agent", "style": "-", "color": "blue", "width": 2},
            {"start": "Role", "end": "Job Matching Agent", "style": "-", "color": "blue", "width": 2},
            
            {"start": "Resume", "end": "Bias Classification Agent", "style": "-", "color": "orange", "width": 2},
            {"start": "Job Description", "end": "Bias Classification Agent", "style": "-", "color": "orange", "width": 2},
            {"start": "Interview Transcript", "end": "Bias Classification Agent", "style": "-", "color": "orange", "width": 2},
            {"start": "Role", "end": "Bias Classification Agent", "style": "-", "color": "orange", "width": 2},
            
            # API Management to LLM
            {"start": "API Key Manager", "end": "Google Gemma 3\n(27B-IT)", "style": "-", "color": "red", "width": 2},
            {"start": "Rate Limiter", "end": "Google Gemma 3\n(27B-IT)", "style": "-", "color": "red", "width": 2},
            
            # Agents to LLM
            {"start": "Job Matching Agent", "end": "Google Gemma 3\n(27B-IT)", "style": "-", "color": "blue", "width": 2},
            {"start": "Bias Classification Agent", "end": "Google Gemma 3\n(27B-IT)", "style": "-", "color": "orange", "width": 2},
            
            # LLM to LangGraph Core
            {"start": "Google Gemma 3\n(27B-IT)", "end": "LangGraph StateGraph", "style": "-", "color": "purple", "width": 2},
            {"start": "LangGraph StateGraph", "end": "InMemorySaver\nCheckpointer", "style": "-", "color": "purple", "width": 2},
            {"start": "LangGraph StateGraph", "end": "Conditional Routing\n(should_continue)", "style": "-", "color": "purple", "width": 2},
            
            # State Management
            {"start": "LangGraph StateGraph", "end": "HiringState\nManagement", "style": "-", "color": "purple", "width": 2},
            {"start": "HiringState\nManagement", "end": "Evaluation Insights\nTracking", "style": "-", "color": "green", "width": 2},
            {"start": "HiringState\nManagement", "end": "Re-evaluation\nCounter", "style": "-", "color": "green", "width": 2},
            
            # Outputs
            {"start": "HiringState\nManagement", "end": "Final Decision", "style": "-", "color": "gray", "width": 2},
            {"start": "HiringState\nManagement", "end": "Bias Classification", "style": "-", "color": "gray", "width": 2},
            {"start": "HiringState\nManagement", "end": "Audit Trail", "style": "-", "color": "gray", "width": 2},
            {"start": "HiringState\nManagement", "end": "Process Metadata", "style": "-", "color": "gray", "width": 2}
        ]
        
        for conn in connections:
            x1, y1 = components[conn["start"]]
            x2, y2 = components[conn["end"]]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=conn["width"], 
                                     color=conn["color"], alpha=0.7,
                                     linestyle=conn["style"]))
        
        # Adjust axis limits for the larger layout
        ax.set_xlim(-1, 13)
        ax.set_ylim(-0.5, 14.5)
        ax.set_title("Multi-Agent AI Hiring System - Complete Architecture", 
                    fontsize=22, fontweight='bold', pad=30)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"🏗️ Enhanced architecture diagram saved to: {save_path}")
        plt.close()
    
    def analyze_correction_performance(self, results_file: str = "results/json/batch_results.json"):
        """Analyze and display detailed correction performance metrics."""
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            if not results:
                print("No results found in file")
                return
            
            df = pd.DataFrame(results)
            
            if 'ground_truth_bias' not in df.columns:
                print("Ground truth data not available for correction analysis")
                return
            
            print("=" * 80)
            print("🔧 CORRECTION PERFORMANCE ANALYSIS")
            print("=" * 80)
            
            # Find correction cases
            correction_cases = df[
                (df['ground_truth_bias'] == 'biased') &  # Original was biased
                (df['bias_classification'] == 'unbiased') &  # System corrected to unbiased
                (df['re_evaluation_count'] > 0)  # Re-evaluation occurred
            ]
            
            total_biased_cases = (df['ground_truth_bias'] == 'biased').sum()
            successful_corrections = len(correction_cases)
            correction_rate = successful_corrections / total_biased_cases if total_biased_cases > 0 else 0
            
            print(f"📊 CORRECTION STATISTICS:")
            print(f"  • Total Biased Cases in Dataset: {total_biased_cases}")
            print(f"  • Successfully Corrected Cases: {successful_corrections}")
            print(f"  • Correction Rate: {correction_rate:.1%}")
            print()
            
            if successful_corrections > 0:
                print(f"🎯 SUCCESSFUL CORRECTION EXAMPLES:")
                print("-" * 50)
                
                for idx, case in correction_cases.iterrows():
                    print(f"📋 Candidate: {case['candidate_id']} ({case['role']})")
                    print(f"  • Ground Truth: {case['ground_truth_decision']} (biased)")
                    print(f"  • Final Decision: {case['final_decision']} (unbiased)")
                    print(f"  • Re-evaluations: {case['re_evaluation_count']}")
                    print()
            
            # Also show cases where bias was detected but not fully corrected
            persistent_bias_cases = df[
                (df['ground_truth_bias'] == 'biased') &  # Original was biased
                (df['bias_classification'] == 'biased')  # System still detected bias
            ]
            
            if len(persistent_bias_cases) > 0:
                print(f"⚠️  PERSISTENT BIAS CASES ({len(persistent_bias_cases)}):")
                print("-" * 50)
                for idx, case in persistent_bias_cases.iterrows():
                    print(f"📋 Candidate: {case['candidate_id']} ({case['role']})")
                    print(f"  • Decision: {case['final_decision']}")
                    print(f"  • Re-evaluations: {case['re_evaluation_count']}")
                    print()
            
            print("=" * 80)
            
        except Exception as e:
            print(f"Error analyzing correction performance: {e}")
    
    def create_all_charts(self, results_file: str = "results/json/batch_results.json"):
        """Create all available charts."""
        print("🎨 Creating all visualization charts...")
        
        # Main evaluation charts (enhanced 6-panel layout)
        self.create_evaluation_charts(results_file, "results/images/evaluation_results.png")
        
        # Enhanced workflow diagram
        self.create_workflow_diagram("results/images/workflow_diagram.png")
        
        # Enhanced system architecture
        self.create_system_architecture_diagram("results/images/system_architecture.png")
        
        # Show correction analysis
        self.analyze_correction_performance(results_file)
        
        print("✅ All charts created successfully!")
        print("📋 Generated charts:")
        print("   • evaluation_results.png - Enhanced 6-panel evaluation with equations and numbers")
        print("   • workflow_diagram.png - Enhanced LangGraph workflow visualization")
        print("   • system_architecture.png - Comprehensive technical architecture diagram")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Generate charts from hiring system results (creates all charts by default)")
    parser.add_argument('--results', default='results/json/batch_results.json',
                       help='Path to results JSON file (default: results/json/batch_results.json)')
    parser.add_argument('--output', default='results/images/evaluation_results.png',
                       help='Output file for evaluation charts when using --evaluation-only (default: results/images/evaluation_results.png)')
    parser.add_argument('--all', action='store_true',
                       help='Create all charts (evaluation, workflow, architecture) - this is the default behavior')
    parser.add_argument('--evaluation-only', action='store_true',
                       help='Create evaluation charts only')
    parser.add_argument('--workflow', action='store_true',
                       help='Create workflow diagram only')
    parser.add_argument('--architecture', action='store_true',
                       help='Create architecture diagram only')
    
    args = parser.parse_args()
    
    generator = HiringSystemChartGenerator()
    
    if args.evaluation_only:
        generator.create_evaluation_charts(args.results, args.output)
    elif args.workflow:
        generator.create_workflow_diagram()
    elif args.architecture:
        generator.create_system_architecture_diagram()
    else:
        # Default behavior: create all charts
        generator.create_all_charts(args.results)


if __name__ == "__main__":
    main()
