"""
Generate Presentation Visuals
Creates custom graphs and plots for the presentation slides
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Create presentation_visuals directory
Path("presentation_visuals").mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

def generate_slide_2_accuracy_comparison():
    """
    Slide 2: Accuracy comparison graphic (33% → 48% → 65%)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    categories = ['Random\nGuessing', 'Naive Baseline\n(Always Home Win)', 'Our Target', 'Our Result']
    accuracies = [33.3, 47.9, 65.0, 68.2]
    colors = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']
    
    bars = ax.barh(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(acc + 1, i, f'{acc}%', va='center', fontweight='bold', fontsize=14)
    
    # Add arrow showing progression
    ax.annotate('', xy=(68.2, 3), xytext=(33.3, 0),
                arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.3))
    
    ax.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance: Beating Baselines', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0, 80])
    ax.grid(axis='x', alpha=0.3)
    
    # Add annotations
    ax.text(33.3, -0.3, '❌', fontsize=20, ha='center')
    ax.text(47.9, 0.7, '⚠️', fontsize=20, ha='center')
    ax.text(65.0, 1.7, '🎯', fontsize=20, ha='center')
    ax.text(68.2, 2.7, '✅', fontsize=20, ha='center')
    
    plt.tight_layout()
    plt.savefig('presentation_visuals/presentation_slide_2.png', dpi=300, bbox_inches='tight')
    print("✓ Slide 2: Accuracy comparison saved")
    plt.close()

def generate_slide_4_exponential_decay():
    """
    Slide 4: Exponential decay weighting curve
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate decay weights
    n_matches = 10
    x = np.arange(1, n_matches + 1)
    weights = np.exp(np.linspace(-1, 0, n_matches))
    weights_normalized = weights / weights.sum()
    
    # Plot
    ax.plot(x, weights_normalized, 'o-', linewidth=3, markersize=10, 
            color='#3498DB', label='Exponential Decay Weight')
    ax.fill_between(x, weights_normalized, alpha=0.3, color='#3498DB')
    
    # Highlight most recent vs oldest
    ax.plot(1, weights_normalized[0], 'o', markersize=15, color='#2ECC71', 
            label=f'Most Recent: {weights_normalized[0]:.3f}', zorder=5)
    ax.plot(10, weights_normalized[-1], 'o', markersize=15, color='#E74C3C', 
            label=f'Oldest: {weights_normalized[-1]:.3f}', zorder=5)
    
    ax.set_xlabel('Match Number (1 = Most Recent)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weight', fontsize=14, fontweight='bold')
    ax.set_title('Recency Weighting: Recent Matches Count More!\n(Key Innovation: +4.2% Accuracy Boost)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in x])
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.annotate('Recent form\nmatters more!', xy=(2, weights_normalized[1]), 
                xytext=(4, weights_normalized[1] + 0.03),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                fontsize=12, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig('presentation_visuals/presentation_slide_4.png', dpi=300, bbox_inches='tight')
    print("✓ Slide 4: Exponential decay curve saved")
    plt.close()

def generate_slide_5_temporal_split():
    """
    Slide 5: Temporal train/test split timeline
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Timeline
    train_width = 0.8
    test_width = 0.2
    
    # Training set
    ax.barh(0, train_width, height=0.4, left=0, color='#3498DB', alpha=0.7, 
            edgecolor='black', linewidth=2, label='Training Set')
    ax.text(train_width/2, 0, 'TRAINING SET\n290 matches\n2019-2022 seasons', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    
    # Test set
    ax.barh(0, test_width, height=0.4, left=train_width, color='#F39C12', alpha=0.7, 
            edgecolor='black', linewidth=2, label='Test Set')
    ax.text(train_width + test_width/2, 0, 'TEST SET\n145 matches\n2022-2023', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrow showing time progression
    ax.annotate('', xy=(0.95, -0.35), xytext=(0.05, -0.35),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    ax.text(0.5, -0.45, 'Time Direction →', ha='center', fontsize=13, fontweight='bold')
    
    # Add annotations
    ax.text(0.5, 0.6, '✓ Train on past data', ha='center', fontsize=13, 
            bbox=dict(boxstyle='round', facecolor='#2ECC71', alpha=0.3))
    ax.text(0.5, 0.75, '✓ Test on future data', ha='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='#2ECC71', alpha=0.3))
    ax.text(0.5, 0.9, '✓ No Data Leakage!', ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#2ECC71', alpha=0.5))
    
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.5, 1])
    ax.set_title('Temporal Train/Test Split: Proper Validation Strategy\n(80% Training / 20% Test)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('presentation_visuals/presentation_slide_5.png', dpi=300, bbox_inches='tight')
    print("✓ Slide 5: Temporal split timeline saved")
    plt.close()

def generate_slide_9_literature_comparison():
    """
    Slide 9: Literature comparison chart
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Literature range
    lit_min, lit_max = 60, 80
    our_result = 68.2
    
    # Draw range bar
    ax.barh(0, lit_max - lit_min, left=lit_min, height=0.3, 
            color='#BDC3C7', alpha=0.5, edgecolor='black', linewidth=2,
            label='Published Literature Range')
    
    # Mark key points
    ax.plot([lit_min, lit_max], [0, 0], 'o', markersize=12, color='gray')
    ax.text(lit_min, -0.15, f'{lit_min}%\n(Lower)', ha='center', fontsize=11, fontweight='bold')
    ax.text(lit_max, -0.15, f'{lit_max}%\n(Upper)', ha='center', fontsize=11, fontweight='bold')
    ax.text(70, 0.15, 'Typical Range:\nLogistic Regression\n60-80%', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))
    
    # Our result
    ax.plot(our_result, 0, '*', markersize=35, color='#2ECC71', 
            markeredgecolor='black', markeredgewidth=2, label='Our Model', zorder=10)
    ax.text(our_result, 0.45, f'OUR MODEL\n{our_result}%', ha='center', fontsize=14, 
            fontweight='bold', color='#2ECC71',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#2ECC71', linewidth=3))
    
    # Add arrow
    ax.annotate('', xy=(our_result, 0.35), xytext=(our_result, 0.05),
                arrowprops=dict(arrowstyle='->', lw=3, color='#2ECC71'))
    
    # Add achievement badges
    ax.text(50, 0.55, '✓ Upper Tier Performance', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#2ECC71', alpha=0.3))
    ax.text(50, 0.7, '✓ Exceeds 65% Target', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#3498DB', alpha=0.3))
    
    ax.set_xlim([55, 85])
    ax.set_ylim([-0.3, 0.8])
    ax.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('How We Compare with Published Research\nLogistic Regression for Soccer Prediction', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('presentation_visuals/presentation_slide_9.png', dpi=300, bbox_inches='tight')
    print("✓ Slide 9: Literature comparison saved")
    plt.close()

def generate_slide_10_summary_dashboard():
    """
    Slide 10: Summary dashboard with key metrics
    """
    fig = plt.figure(figsize=(14, 8))
    
    # Create grid for 4 metric cards
    metrics = [
        {'value': '68.2%', 'label': 'Test Accuracy', 'icon': '🏆', 'color': '#2ECC71'},
        {'value': '+42%', 'label': 'vs Baseline', 'icon': '📈', 'color': '#3498DB'},
        {'value': '+4.2pp', 'label': 'Recency Boost', 'icon': '⭐', 'color': '#F39C12'},
        {'value': '[62.8%, 73.1%]', 'label': '95% Confidence Interval', 'icon': '✓', 'color': '#9B59B6'}
    ]
    
    for i, metric in enumerate(metrics):
        ax = plt.subplot(2, 2, i+1)
        
        # Create card background
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=metric['color'], 
                                   alpha=0.15, edgecolor=metric['color'], linewidth=4))
        
        # Add icon
        ax.text(0.5, 0.75, metric['icon'], ha='center', va='center', fontsize=50)
        
        # Add value
        ax.text(0.5, 0.45, metric['value'], ha='center', va='center', 
                fontsize=32, fontweight='bold', color=metric['color'])
        
        # Add label
        ax.text(0.5, 0.2, metric['label'], ha='center', va='center', 
                fontsize=14, fontweight='bold', wrap=True)
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axis('off')
    
    fig.suptitle('Champions League Prediction: Key Achievements', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Add subtitle
    fig.text(0.5, 0.02, 'Multinomial Logistic Regression with Recency-Weighted Features', 
             ha='center', fontsize=12, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig('presentation_visuals/presentation_slide_10.png', dpi=300, bbox_inches='tight')
    print("✓ Slide 10: Summary dashboard saved")
    plt.close()

def generate_slide_6_large_accuracy():
    """
    Slide 6: Large accuracy display for presentation
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Main accuracy number
    ax.text(0.5, 0.6, '68.2%', ha='center', va='center', fontsize=120, 
            fontweight='bold', color='#2ECC71')
    ax.text(0.5, 0.45, 'Test Accuracy', ha='center', va='center', fontsize=24, 
            style='italic', color='#34495E')
    
    # Comparison boxes
    boxes = [
        {'y': 0.25, 'text': 'Baseline: 47.9%', 'color': '#E74C3C'},
        {'y': 0.15, 'text': 'Improvement: +42%', 'color': '#3498DB'},
        {'y': 0.05, 'text': 'Target: 65% ✓ EXCEEDED', 'color': '#2ECC71'}
    ]
    
    for box in boxes:
        ax.text(0.5, box['y'], box['text'], ha='center', va='center', fontsize=18,
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.8', 
                facecolor=box['color'], alpha=0.3, edgecolor=box['color'], linewidth=2))
    
    # Trophy
    ax.text(0.85, 0.6, '🏆', fontsize=80)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('presentation_visuals/presentation_slide_6_large.png', dpi=300, bbox_inches='tight')
    print("✓ Slide 6: Large accuracy display saved")
    plt.close()

def main():
    """
    Generate all presentation visuals
    """
    print("="*60)
    print("GENERATING PRESENTATION VISUALS")
    print("="*60)
    print()
    
    generate_slide_2_accuracy_comparison()
    generate_slide_4_exponential_decay()
    generate_slide_5_temporal_split()
    generate_slide_6_large_accuracy()
    generate_slide_9_literature_comparison()
    generate_slide_10_summary_dashboard()
    
    print()
    print("="*60)
    print("✓ ALL PRESENTATION VISUALS GENERATED!")
    print("="*60)
    print("\nFiles saved to: presentation_visuals/")
    print("\nGenerated files:")
    print("  - presentation_slide_2.png  (Accuracy comparison)")
    print("  - presentation_slide_4.png  (Exponential decay curve)")
    print("  - presentation_slide_5.png  (Temporal train/test split)")
    print("  - presentation_slide_6_large.png  (Large accuracy display)")
    print("  - presentation_slide_9.png  (Literature comparison)")
    print("  - presentation_slide_10.png (Summary dashboard)")
    print("\nUse these in your presentation along with:")
    print("  - results/confusion_matrix.png")
    print("  - results/feature_importance.png")
    print("  - results/calibration_curves.png")
    print("  - results/confidence_analysis.png")

if __name__ == "__main__":
    main()

