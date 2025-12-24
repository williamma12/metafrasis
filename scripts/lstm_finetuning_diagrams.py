import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 10))

# Color scheme
lstm_color = '#E8F4F8'
adapter_color = '#FFE5CC'
lora_color = '#E5CCF5'
head_color = '#CCF5E5'
frozen_color = '#E0E0E0'

def draw_lstm_cell(ax, x, y, width, height, label, frozen=False):
    """Draw an LSTM cell"""
    color = frozen_color if frozen else lstm_color
    rect = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, 
            ha='center', va='center', fontsize=10, weight='bold')

def draw_adapter(ax, x, y, width, height, label):
    """Draw an adapter module"""
    rect = FancyBboxPatch((x, y), width, height, 
                          boxstyle="round,pad=0.03", 
                          edgecolor='red', facecolor=adapter_color, 
                          linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, 
            ha='center', va='center', fontsize=9, weight='bold', color='red')

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=20, 
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y, label, fontsize=8)

# ============= DIAGRAM 1: Standard LSTM with Task Head =============
ax1 = axes[0]
ax1.set_xlim(0, 5)
ax1.set_ylim(0, 10)
ax1.axis('off')
ax1.set_title('Standard Fine-tuning:\nTask-Specific Head', fontsize=14, weight='bold', pad=20)

# Input
ax1.text(2.5, 0.5, 'Input Sequence', ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# LSTM layers (frozen)
draw_lstm_cell(ax1, 1.5, 2, 2, 0.8, 'LSTM Layer 1', frozen=True)
draw_lstm_cell(ax1, 1.5, 3.5, 2, 0.8, 'LSTM Layer 2', frozen=True)
draw_lstm_cell(ax1, 1.5, 5, 2, 0.8, 'LSTM Layer 3', frozen=True)

# Task head
rect = FancyBboxPatch((1.5, 7), 2, 1.5, 
                      boxstyle="round,pad=0.05", 
                      edgecolor='green', facecolor=head_color, linewidth=3)
ax1.add_patch(rect)
ax1.text(2.5, 7.3, 'Task Head', ha='center', fontsize=10, weight='bold', color='green')
ax1.text(2.5, 7.75, '(Trainable)', ha='center', fontsize=8, style='italic')

# Output
ax1.text(2.5, 9.2, 'Output', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# Arrows
draw_arrow(ax1, 2.5, 0.8, 2.5, 2)
draw_arrow(ax1, 2.5, 2.8, 2.5, 3.5)
draw_arrow(ax1, 2.5, 4.3, 2.5, 5)
draw_arrow(ax1, 2.5, 5.8, 2.5, 7)
draw_arrow(ax1, 2.5, 8.5, 2.5, 9)

# Legend
ax1.text(0.3, 8.5, '🔒 Frozen', fontsize=9)
ax1.text(0.3, 8, '✏️ Trainable', fontsize=9, color='green')

# ============= DIAGRAM 2: LSTM with Adapter Layers =============
ax2 = axes[1]
ax2.set_xlim(0, 5)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('Adapter Fine-tuning:\nBottleneck Adapters', fontsize=14, weight='bold', pad=20)

# Input
ax2.text(2.5, 0.5, 'Input Sequence', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

layer_y = [2, 4.2, 6.4]
for i, y in enumerate(layer_y):
    # LSTM layer (frozen)
    draw_lstm_cell(ax2, 1.5, y, 2, 0.8, f'LSTM Layer {i+1}', frozen=True)
    
    # Adapter after LSTM layer
    adapter_y = y + 1.2
    draw_adapter(ax2, 1.7, adapter_y, 1.6, 0.6, f'Adapter {i+1}')
    
    # Show residual connection
    if i < len(layer_y):
        # Arrow from LSTM to adapter
        draw_arrow(ax2, 2.5, y + 0.8, 2.5, adapter_y)
        
        # Residual arrow (curved)
        arc = mpatches.FancyBboxPatch((0.8, y + 0.4), 0.5, adapter_y - y + 0.2,
                                      boxstyle="round,pad=0.02",
                                      edgecolor='blue', facecolor='none',
                                      linewidth=1.5, linestyle='--')
        ax2.add_patch(arc)
        ax2.text(0.4, y + 1, '+', fontsize=14, color='blue', weight='bold')
        
        if i < len(layer_y) - 1:
            draw_arrow(ax2, 2.5, adapter_y + 0.6, 2.5, layer_y[i+1])

# Output
draw_arrow(ax2, 2.5, 8.2, 2.5, 9)
ax2.text(2.5, 9.2, 'Output', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# Adapter detail (zoomed)
ax2.text(4.2, 5.5, 'Adapter Detail:', fontsize=9, weight='bold')
ax2.text(4.2, 5.1, '↓ Down (512→64)', fontsize=7)
ax2.text(4.2, 4.8, 'ReLU', fontsize=7)
ax2.text(4.2, 4.5, '↑ Up (64→512)', fontsize=7)
ax2.text(4.2, 4.2, '+ Residual', fontsize=7, color='blue')

# Legend
ax2.text(0.3, 8.5, '🔒 Frozen', fontsize=9)
ax2.text(0.3, 8, '✏️ Trainable', fontsize=9, color='red')

# ============= DIAGRAM 3: LSTM with LoRA =============
ax3 = axes[2]
ax3.set_xlim(0, 6)
ax3.set_ylim(0, 10)
ax3.axis('off')
ax3.set_title('LoRA Fine-tuning:\nLow-Rank Adaptation', fontsize=14, weight='bold', pad=20)

# Input
ax3.text(3, 0.5, 'Input (x_t)', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# Single LSTM cell with LoRA detail
lstm_y = 3
draw_lstm_cell(ax3, 1.8, lstm_y, 2.4, 1.2, 'LSTM Cell', frozen=True)

# LoRA components on the side
lora_x = 4.5
lora_y = lstm_y + 0.2

# Show weight matrix decomposition
ax3.text(1.2, lstm_y + 0.6, 'W', fontsize=11, weight='bold', 
         bbox=dict(boxstyle='circle', facecolor=frozen_color, edgecolor='black'))
ax3.text(0.5, lstm_y + 0.6, '🔒', fontsize=10)

# LoRA matrices
rect_A = FancyBboxPatch((lora_x, lora_y), 0.8, 0.3, 
                        edgecolor='purple', facecolor=lora_color, linewidth=2)
ax3.add_patch(rect_A)
ax3.text(lora_x + 0.4, lora_y + 0.15, 'A', ha='center', fontsize=10, weight='bold')

rect_B = FancyBboxPatch((lora_x, lora_y + 0.5), 0.8, 0.3, 
                        edgecolor='purple', facecolor=lora_color, linewidth=2)
ax3.add_patch(rect_B)
ax3.text(lora_x + 0.4, lora_y + 0.65, 'B', ha='center', fontsize=10, weight='bold')

# Show the addition
ax3.text(1.4, lstm_y + 1.5, "W' = W + BA", fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='purple', linewidth=2))

# Gate structure inside LSTM
gate_x = 2.3
gate_y = lstm_y + 0.2
gates = ['i', 'f', 'g', 'o']
for i, gate in enumerate(gates):
    small_x = gate_x + (i % 2) * 0.5
    small_y = gate_y + (i // 2) * 0.3
    circ = plt.Circle((small_x, small_y), 0.15, color='white', ec='black', linewidth=1)
    ax3.add_patch(circ)
    ax3.text(small_x, small_y, gate, ha='center', va='center', fontsize=8)

# Show LoRA applied to each gate
ax3.annotate('', xy=(lora_x, lora_y + 0.4), xytext=(gate_x + 0.6, gate_y + 0.3),
            arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, linestyle='--'))
ax3.text(4, lstm_y + 1.2, 'LoRA\nAdapter', fontsize=8, color='purple', 
         ha='center', weight='bold')

# Hidden state and cell state
draw_arrow(ax3, 3, lstm_y + 1.2, 3, lstm_y + 2)
ax3.text(3, lstm_y + 2.3, 'h_t, c_t', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# Show stacking
ax3.text(3, lstm_y + 3.2, '⋮', ha='center', fontsize=20)
ax3.text(3, lstm_y + 3.8, 'Stack L layers', ha='center', fontsize=9, style='italic')

# Output
ax3.text(3, 9.2, 'Output', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# LoRA math detail
ax3.text(0.3, 1.5, 'LoRA Math:', fontsize=9, weight='bold')
ax3.text(0.3, 1.1, 'W: 512×512', fontsize=7)
ax3.text(0.3, 0.8, 'A: 8×512', fontsize=7, color='purple')
ax3.text(0.3, 0.5, 'B: 512×8', fontsize=7, color='purple')
ax3.text(0.3, 0.2, '~97% ↓ params', fontsize=7, color='green', weight='bold')

# Legend
ax3.text(0.3, 8.5, '🔒 Frozen', fontsize=9)
ax3.text(0.3, 8, '✏️ Trainable', fontsize=9, color='purple')

plt.tight_layout()
plt.savefig('/tmp/lstm_adapter_architectures.png', dpi=300, bbox_inches='tight', 
            facecolor='white')
plt.show()

print("Diagram saved! Here's what each shows:")
print("\n1. STANDARD FINE-TUNING:")
print("   - Freeze all LSTM layers")
print("   - Only train the task-specific head on top")
print("   - Simplest approach, but limited adaptation")

print("\n2. ADAPTER LAYERS:")
print("   - Insert small bottleneck modules after each LSTM layer")
print("   - Residual connection preserves original activations")
print("   - Down-project (512→64), activate, up-project (64→512)")
print("   - Only ~3-5% of original parameters")

print("\n3. LoRA:")
print("   - Modify weight matrices with low-rank decomposition")
print("   - W' = W + BA (frozen W + trainable low-rank BA)")
print("   - Applied to all 4 LSTM gates (i, f, g, o)")
print("   - Most parameter-efficient: ~2-3% of original")

print("\nKey differences:")
print("- Adapters: Sequential modules added to architecture")
print("- LoRA: Parallel updates to existing weight matrices")
print("- Both preserve pre-trained knowledge while adapting efficiently")
