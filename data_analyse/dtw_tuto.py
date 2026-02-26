import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw
from scipy.stats import pearsonr
import pandas as pd
# https://medium.com/@markstent/dynamic-time-warping-a8c5027defb6

# Generate dummy data for the time series
n_points = 100
x = np.linspace(0, 4*np.pi, n_points)

time_series_a = np.sin(x)
time_series_b = np.cos(x)

# Calculate DTW distance (minimum cumulative distance) and obtain the warping paths
distance, paths = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
best_path = dtw.best_path(paths)
similarity_score = distance / len(best_path)

# ============== WARPING FUNCTIONS ==============*7
def warp_series_b_to_a(series_a, series_b, path):
    """
    Warp series B to align with series A
    Returns a warped version of B with same length as A
    """
    warped_b = np.zeros(len(series_a))
    
    # Create mapping from A indices to B indices
    a_to_b_mapping = {}
    for a_idx, b_idx in path:
        if a_idx not in a_to_b_mapping:
            a_to_b_mapping[a_idx] = []
        a_to_b_mapping[a_idx].append(b_idx)
    
    # Map values from B to A using the warping path
    for a_idx in range(len(series_a)):
        if a_idx in a_to_b_mapping:
            # Average if multiple B indices map to same A index
            b_indices = a_to_b_mapping[a_idx]
            warped_b[a_idx] = np.mean([series_b[b_idx] for b_idx in b_indices])
    
    return warped_b

def warp_series_a_to_b(series_a, series_b, path):
    """
    Warp series A to align with series B
    Returns a warped version of A with same length as B
    """
    warped_a = np.zeros(len(series_b))
    
    # Create mapping from B indices to A indices
    b_to_a_mapping = {}
    for a_idx, b_idx in path:
        if b_idx not in b_to_a_mapping:
            b_to_a_mapping[b_idx] = []
        b_to_a_mapping[b_idx].append(a_idx)
    
    # Map values from A to B using the warping path
    for b_idx in range(len(series_b)):
        if b_idx in b_to_a_mapping:
            a_indices = b_to_a_mapping[b_idx]
            warped_a[b_idx] = np.mean([series_a[a_idx] for a_idx in a_indices])
    
    return warped_a

# ============== APPLY WARPING ==============
warped_b_to_a = warp_series_b_to_a(time_series_a, time_series_b, best_path)
warped_a_to_b = warp_series_a_to_b(time_series_a, time_series_b, best_path)

# Calculate correlations
correlation_original = pearsonr(time_series_a, time_series_b)[0]
correlation_after_warp_b = pearsonr(time_series_a, warped_b_to_a)[0]
correlation_after_warp_a = pearsonr(warped_a_to_b, time_series_b)[0]

# Calculate point-wise errors
mse_original = np.mean((time_series_a - time_series_b)**2)
mse_warped = np.mean((time_series_a - warped_b_to_a)**2)

# ============== VISUALIZATION ==============
fig = plt.figure(figsize=(16, 12))

# 1. Original Time Series Plot
ax1 = fig.add_subplot(3,3,1)
ax1.plot(time_series_a, label='Sin (Series A)', color='blue', linewidth=2)
ax1.plot(time_series_b, label='Cos (Series B)', linestyle='--', color='orange', linewidth=2)
ax1.set_title('Original Time Series', fontsize=12, fontweight='bold')
ax1.set_xlabel('Index')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(True, alpha=0.3)

# visualize warping path
fig2 = plt.figure(figsize=(16, 12))
dtwvis.plot_warpingpaths(time_series_a, time_series_b,paths,path=best_path,figure=fig2,showlegend=True,shownumbers=False,show_diagonal=True)

# 3. Statistics Summary
ax3 = fig.add_subplot(3,3,2)
ax3.axis('off')
stats_text = f"""
DTW Statistics:

DTW Distance: {distance:.4f}
Similarity Score: {similarity_score:.4f}
Path Length: {len(best_path)}

Correlations:
Original: {correlation_original:.4f}
After Warp (B→A): {correlation_after_warp_b:.4f}
After Warp (A→B): {correlation_after_warp_a:.4f}

MSE:
Original: {mse_original:.4f}
After Warp: {mse_warped:.4f}
Improvement: {((mse_original-mse_warped)/mse_original*100):.1f}%
"""
ax3.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Series B Warped to Match Series A
ax4 = fig.add_subplot(3,3,4)
ax4.plot(time_series_a, label='Sin (Series A)', color='blue', marker='o', markersize=3, linewidth=2, alpha=0.7)
ax4.plot(warped_b_to_a, label='Cos Warped to A', color='red', marker='s', markersize=2, linewidth=2, alpha=0.7)
ax4.set_title(f'Series B Warped to Match A\nCorrelation: {correlation_after_warp_b:.4f}', 
              fontsize=12, fontweight='bold')
ax4.set_xlabel('Index (Series A length)')
ax4.set_ylabel('Value')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Series A Warped to Match Series B
ax5 = fig.add_subplot(3, 3, 5)
ax5.plot(warped_a_to_b, label='Sin Warped to B', color='blue', marker='s', markersize=2, linewidth=2, alpha=0.7)
ax5.plot(time_series_b, label='Cos (Series B)', color='orange', marker='x', markersize=3, linewidth=2, alpha=0.7)
ax5.set_title(f'Series A Warped to Match B\nCorrelation: {correlation_after_warp_a:.4f}', 
              fontsize=12, fontweight='bold')
ax5.set_xlabel('Index (Series B length)')
ax5.set_ylabel('Value')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Overlay Comparison Before Warping
ax6 = fig.add_subplot(3, 3, 6)
ax6.plot(time_series_a, label='Sin (A)', color='blue', linewidth=2, alpha=0.7)
ax6.plot(time_series_b, label='Cos (B)', color='orange', linewidth=2, alpha=0.7)
# Show phase difference
ax6.axvline(x=25, color='red', linestyle=':', alpha=0.5, label='Phase Diff')
ax6.set_title('Before Warping (Phase Shifted)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Index')
ax6.set_ylabel('Value')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Point-to-Point Comparison After DTW Alignment
ax7 = fig.add_subplot(3, 3, 7)
sample_indices = range(0, len(best_path), max(1, len(best_path)//50))  # Sample points for clarity
for idx in sample_indices:
    a, b = best_path[idx]
    ax7.plot([a, b], [time_series_a[a], time_series_b[b]], 
             color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
ax7.plot(time_series_a, label='Sin (A)', color='blue', marker='o', markersize=4, linewidth=1.5)
ax7.plot(time_series_b, label='Cos (B)', color='orange', marker='x', markersize=4, linestyle='--', linewidth=1.5)
ax7.set_title('Point-to-Point DTW Alignment', fontsize=12, fontweight='bold')
ax7.set_xlabel('Index')
ax7.set_ylabel('Value')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Direct Comparison After Warping (with difference lines)
ax8 = fig.add_subplot(3, 3, 8)
indices = range(len(time_series_a))
ax8.plot(indices, time_series_a, label='Sin (A)', color='blue', marker='o', markersize=3, linewidth=2, alpha=0.7)
ax8.plot(indices, warped_b_to_a, label='Cos (Warped)', color='red', marker='s', markersize=2, linewidth=2, alpha=0.7)
# Show differences every 10 points for clarity
for i in indices[::10]:
    ax8.plot([i, i], [time_series_a[i], warped_b_to_a[i]], 
             color='grey', linestyle='--', linewidth=1, alpha=0.6)
ax8.set_title('After Warping (Aligned)', fontsize=12, fontweight='bold')
ax8.set_xlabel('Index')
ax8.set_ylabel('Value')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Error Visualization
ax9 = fig.add_subplot(3, 3, 9)
error_before = np.abs(time_series_a - time_series_b)
error_after = np.abs(time_series_a - warped_b_to_a)
ax9.plot(error_before, label='Error Before Warp', color='orange', linewidth=2, alpha=0.7)
ax9.plot(error_after, label='Error After Warp', color='green', linewidth=2, alpha=0.7)
ax9.fill_between(range(len(error_before)), error_before, alpha=0.3, color='orange')
ax9.fill_between(range(len(error_after)), error_after, alpha=0.3, color='green')
ax9.set_title('Absolute Error Comparison', fontsize=12, fontweight='bold')
ax9.set_xlabel('Index')
ax9.set_ylabel('Absolute Error')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============== RESULTS DATAFRAME ==============
results_df = pd.DataFrame({
    'Metric': [
        'DTW Distance',
        'DTW Similarity Score',
        'Warping Path Length',
        'Correlation (Original)',
        'Correlation (B warped to A)',
        'Correlation (A warped to B)',
        'MSE (Original)',
        'MSE (After Warping)',
        'Error Reduction (%)'
    ],
    'Value': [
        f'{distance:.4f}',
        f'{similarity_score:.4f}',
        len(best_path),
        f'{correlation_original:.4f}',
        f'{correlation_after_warp_b:.4f}',
        f'{correlation_after_warp_a:.4f}',
        f'{mse_original:.4f}',
        f'{mse_warped:.4f}',
        f'{((mse_original-mse_warped)/mse_original*100):.2f}%'
    ],
    'Description': [
        'Total accumulated distance along optimal warping path',
        'Lower scores indicate greater similarity',
        'Number of alignment points in the warping path',
        'Pearson correlation between original sin and cos',
        'Correlation after warping cos to match sin',
        'Correlation after warping sin to match cos',
        'Mean squared error before warping',
        'Mean squared error after warping',
        'Percentage improvement in alignment error'
    ]
})

print("\n" + "="*80)
print("DTW ANALYSIS RESULTS: SIN vs COS")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

