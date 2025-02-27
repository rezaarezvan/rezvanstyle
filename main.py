import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec


def save_stylesheets():
    os.makedirs('styles', exist_ok=True)

    with open('styles/paper_light.mplstyle', 'w') as f:
        f.write("""
# Figure properties
figure.figsize: 8, 6
figure.dpi: 300
figure.facecolor: white
figure.edgecolor: white

# Font properties
font.family: serif
font.serif: Computer Modern Roman, DejaVu Serif, Times New Roman
font.size: 12
font.weight: normal
mathtext.fontset: cm

# Axes properties
axes.facecolor: white
axes.edgecolor: black
axes.linewidth: 1.0
axes.grid: True
axes.titlesize: 14
axes.titleweight: bold
axes.labelsize: 12
axes.labelweight: normal
axes.spines.top: True
axes.spines.right: True
axes.spines.left: True
axes.spines.bottom: True
axes.formatter.use_mathtext: True
axes.autolimit_mode: round_numbers
axes.xmargin: 0
axes.ymargin: 0

# Grid properties
grid.color: lightgray
grid.linestyle: --
grid.linewidth: 0.5
grid.alpha: 0.5

# Line properties
lines.linewidth: 2.0
lines.markersize: 8
lines.markeredgewidth: 1.5

# Patch properties
patch.linewidth: 1.0
patch.edgecolor: black
patch.force_edgecolor: True

# Legend properties
legend.frameon: True
legend.framealpha: 0.9
legend.fancybox: True
legend.fontsize: 10
legend.title_fontsize: 12
legend.edgecolor: 0.8
legend.borderpad: 0.4
legend.columnspacing: 1.0
legend.handlelength: 2.0

# Scatter properties
scatter.marker: o
scatter.edgecolors: face

# Color maps and cycles
image.cmap: viridis
axes.prop_cycle: cycler('color', ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf'])

# Ticks
xtick.major.size: 6
xtick.major.width: 1.0
xtick.minor.size: 3
xtick.minor.width: 0.8
xtick.labelsize: 10
xtick.direction: in
xtick.minor.visible: True
xtick.top: True

ytick.major.size: 6
ytick.major.width: 1.0
ytick.minor.size: 3
ytick.minor.width: 0.8
ytick.labelsize: 10
ytick.direction: in
ytick.minor.visible: True
ytick.right: True

# Savefig
savefig.dpi: 300
savefig.format: pdf
savefig.bbox: tight
savefig.pad_inches: 0.1
savefig.transparent: False

# Errorbar
errorbar.capsize: 3
""")

    with open('styles/paper_dark.mplstyle', 'w') as f:
        f.write("""
# Figure properties
figure.figsize: 8, 6
figure.dpi: 300
figure.facecolor: 121212
figure.edgecolor: 121212

# Font properties
font.family: serif
font.serif: Computer Modern Roman, DejaVu Serif, Times New Roman
font.size: 12
font.weight: normal
mathtext.fontset: cm

# Text properties
text.color: E0E0E0

# Axes properties
axes.facecolor: 121212
axes.edgecolor: E0E0E0
axes.linewidth: 1.0
axes.grid: True
axes.titlesize: 14
axes.titleweight: bold
axes.labelsize: 12
axes.labelweight: normal
axes.spines.top: True
axes.spines.right: True
axes.spines.left: True
axes.spines.bottom: True
axes.formatter.use_mathtext: True
axes.autolimit_mode: round_numbers
axes.xmargin: 0
axes.ymargin: 0
axes.labelcolor: E0E0E0
axes.titlecolor: E0E0E0

# Grid properties
grid.color: 3A3A3A
grid.linestyle: --
grid.linewidth: 0.5
grid.alpha: 0.7

# Line properties
lines.linewidth: 2.0
lines.markersize: 8
lines.markeredgewidth: 1.5
lines.color: E0E0E0

# Patch properties
patch.linewidth: 1.0
patch.edgecolor: E0E0E0
patch.force_edgecolor: True

# Legend properties
legend.frameon: True
legend.framealpha: 0.8
legend.fancybox: True
legend.fontsize: 10
legend.title_fontsize: 12
legend.edgecolor: 3A3A3A
legend.borderpad: 0.4
legend.columnspacing: 1.0
legend.handlelength: 2.0
legend.facecolor: 1E1E1E
legend.labelcolor: E0E0E0

# Scatter properties
scatter.marker: o
scatter.edgecolors: face

# Color maps and cycles
image.cmap: plasma
axes.prop_cycle: cycler('color', ['56B4E9', 'F0E442', '0072B2', 'D55E00', 'CC79A7', '009E73', 'E69F00', '8da0cb', 'ff9f9b', 'e0bc04'])

# Ticks
xtick.major.size: 6
xtick.major.width: 1.0
xtick.minor.size: 3
xtick.minor.width: 0.8
xtick.labelsize: 10
xtick.direction: in
xtick.minor.visible: True
xtick.top: True
xtick.color: E0E0E0
xtick.labelcolor: E0E0E0

ytick.major.size: 6
ytick.major.width: 1.0
ytick.minor.size: 3
ytick.minor.width: 0.8
ytick.labelsize: 10
ytick.direction: in
ytick.minor.visible: True
ytick.right: True
ytick.color: E0E0E0
ytick.labelcolor: E0E0E0

# Savefig
savefig.dpi: 300
savefig.format: pdf
savefig.bbox: tight
savefig.pad_inches: 0.1
savefig.transparent: False
savefig.facecolor: 121212
savefig.edgecolor: 121212

# Errorbar
errorbar.capsize: 3
""")

    print("Stylesheets saved to 'styles/paper_light.mplstyle' and 'styles/paper_dark.mplstyle'")


def generate_data():
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.exp(-0.1 * x)
    y4 = np.cos(x) * np.exp(-0.1 * x)

    np.random.seed(42069)
    noise = np.random.normal(0, 0.1, size=len(x))
    y1_noisy = y1 + noise

    # For histograms and boxplots
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0.5, 1.2, 1000)
    data3 = np.random.normal(-0.5, 0.8, 1000)

    # For bar charts
    categories = ['A', 'B', 'C', 'D', 'E']
    values1 = np.random.randint(10, 50, size=len(categories))
    values2 = np.random.randint(10, 50, size=len(categories))

    # For heatmap
    heatmap_data = np.random.rand(10, 10)

    # For error bars
    x_err = np.linspace(0, 10, 10)
    y_err = np.sin(x_err)
    yerr = 0.2 * np.random.rand(len(y_err))

    # For contour plot
    x_contour = np.linspace(-3, 3, 100)
    y_contour = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x_contour, y_contour)
    Z = np.exp(-(X**2 + Y**2)) + 0.1 * np.exp(-(X-2)**2 - (Y-2)**2)

    return {
        'line': (x, y1, y2, y3, y4),
        'scatter': (x, y1_noisy),
        'hist': (data1, data2, data3),
        'bar': (categories, values1, values2),
        'heatmap': heatmap_data,
        'error': (x_err, y_err, yerr),
        'contour': (X, Y, Z)
    }


def create_line_plot(ax, data, title="Line Plot"):
    x, y1, y2, y3, y4 = data
    ax.plot(x, y1, label=r'$\sin(x)$')
    ax.plot(x, y2, label=r'$\cos(x)$')
    ax.plot(x, y3, label=r'$\sin(x)e^{-0.1x}$')
    ax.plot(x, y4, label=r'$\cos(x)e^{-0.1x}$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title(title)
    ax.legend()


def create_scatter_plot(ax, data, title="Scatter Plot"):
    x, y_noisy = data
    ax.scatter(x, y_noisy, alpha=0.7, label=r'Noisy $\sin(x)$')
    z = np.polyfit(x, y_noisy, 3)
    p = np.poly1d(z)
    ax.plot(x, p(x), '--', label='Trend', lw=2)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x) + \epsilon$')
    ax.set_title(title)
    ax.legend()


def create_histogram(ax, data, title="Histogram"):
    data1, data2, data3 = data
    ax.hist(data1, bins=30, alpha=0.5, label='Data 1', density=True)
    ax.hist(data2, bins=30, alpha=0.5, label='Data 2', density=True)
    ax.hist(data3, bins=30, alpha=0.5, label='Data 3', density=True)

    x = np.linspace(-4, 4, 100)
    for _, (d, label) in enumerate(zip([data1, data2, data3],
                                       ['Data 1', 'Data 2', 'Data 3'])):
        mu, std = np.mean(d), np.std(d)
        pdf = 1/(std * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*std**2))
        ax.plot(x, pdf, lw=2, label=f'PDF {label}')

    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()


def create_bar_chart(ax, data, title="Bar Chart"):
    categories, values1, values2 = data
    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, values1, width, label='Group 1')
    ax.bar(x + width/2, values2, width, label='Group 2')

    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()


def create_box_plot(ax, data, title="Box Plot"):
    data1, data2, data3 = data

    boxplot_data = [data1, data2, data3]
    ax.boxplot(boxplot_data, patch_artist=True, medianprops={'linewidth': 2})

    for _, (box, color) in enumerate(zip(ax.artists,
                                         ['#1f77b4', '#ff7f0e', '#2ca02c'])):
        box.set_facecolor(color)
        box.set_alpha(0.7)

    ax.set_xlabel('Group')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.set_xticklabels(['Data 1', 'Data 2', 'Data 3'])


def create_heatmap(ax, data, title="Heatmap"):
    im = ax.imshow(data, cmap=plt.cm.viridis)
    ax.set_title(title)

    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels([f'X{i}' for i in range(data.shape[1])])
    ax.set_yticklabels([f'Y{i}' for i in range(data.shape[0])])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            _ = ax.text(j, i, f'{data[i, j]:.2f}',
                        ha="center", va="center", color="w" if data[i, j] > 0.5 else "black",
                        fontsize=8)


def create_error_bar(ax, data, title="Error Bar Plot"):
    x, y, yerr = data
    ax.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, label='Data with Errors')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.legend()


def create_contour_plot(ax, data, title="Contour Plot"):
    X, Y, Z = data

    contourf = ax.contourf(X, Y, Z, 10, cmap='viridis')
    contour = ax.contour(X, Y, Z, 10, colors='white', linewidths=0.5)

    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)

    plt.colorbar(contourf, ax=ax)


def create_subplots(ax, data, title="Multiple Plot Types"):
    gs = GridSpec(2, 2, figure=ax.figure,
                  left=ax.get_position().x0 + 0.05,
                  right=ax.get_position().x1 - 0.05,
                  bottom=ax.get_position().y0 + 0.05,
                  top=ax.get_position().y1 - 0.1)

    ax.axis('off')
    ax.set_title(title)

    x, y1, _, _, _ = data['line']

    ax1 = ax.figure.add_subplot(gs[0, 0])
    ax1.plot(x[:30], y1[:30], lw=2)
    ax1.set_title('Line', fontsize=10)

    ax2 = ax.figure.add_subplot(gs[0, 1])
    x_scatter, y_scatter = data['scatter']
    ax2.scatter(x_scatter[:30], y_scatter[:30], s=20)
    ax2.set_title('Scatter', fontsize=10)

    ax3 = ax.figure.add_subplot(gs[1, 0])
    categories, values1, _ = data['bar']
    ax3.bar(categories[:3], values1[:3], width=0.6)
    ax3.set_title('Bar', fontsize=10)

    ax4 = ax.figure.add_subplot(gs[1, 1])
    data1, _, _ = data['hist']
    ax4.hist(data1[:200], bins=10)
    ax4.set_title('Histogram', fontsize=10)

    for a in [ax1, ax2, ax3, ax4]:
        a.tick_params(labelsize=8)
        for label in a.get_xticklabels():
            label.set_rotation(45)


def create_math_plot(ax, title="Mathematical Functions"):
    x = np.linspace(-5, 5, 1000)

    y1 = np.exp(-x**2) * np.sin(2*np.pi*x)
    y2 = np.exp(-0.5*x**2)
    y3 = np.sin(2*np.pi*x) / (1 + x**2)

    ax.plot(x, y1, label=r'$e^{-x^2} \sin(2\pi x)$')
    ax.plot(x, y2, label=r'$e^{-0.5x^2}$')
    ax.plot(x, y3, label=r'$\frac{\sin(2\pi x)}{1+x^2}$')

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(x=-2, color='gray', linestyle='--', alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title(title)

    ax.text(1.5, 0.7, r'$\mathcal{L} = \int_{-\infty}^{\infty} |f(x)|^2 dx$',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    ax.set_xlim(-3, 3)
    ax.legend(loc='lower right')


def create_ml_plot(ax, title="ML Metrics"):
    fpr = np.linspace(0, 1, 100)

    tpr1 = fpr + (1 - fpr) * np.random.rand(len(fpr))*0.3
    tpr1 = np.minimum(tpr1, 1.0)

    tpr2 = fpr**0.5
    tpr3 = fpr**0.3

    auc1 = np.trapz(tpr1, fpr)
    auc2 = np.trapz(tpr2, fpr)
    auc3 = np.trapz(tpr3, fpr)

    ax.plot(fpr, tpr1, label=f'Model 1 (AUC={auc1:.3f})')
    ax.plot(fpr, tpr2, label=f'Model 2 (AUC={auc2:.3f})')
    ax.plot(fpr, tpr3, label=f'Model 3 (AUC={auc3:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', label='Random')

    ax.grid(True, alpha=0.3)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)

    ax.legend(loc='lower right')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.text(0.6, 0.2, 'Higher AUC = Better Model',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))


def test_styles():
    save_stylesheets()
    data = generate_data()

    for style_name in ['paper_light', 'paper_dark']:
        plt.style.use(f'styles/{style_name}.mplstyle')
        fig = plt.figure(figsize=(15, 20))

        nrows, ncols = 5, 2

        ax1 = fig.add_subplot(nrows, ncols, 1)
        create_line_plot(ax1, data['line'], title="Line Plot")

        ax2 = fig.add_subplot(nrows, ncols, 2)
        create_scatter_plot(ax2, data['scatter'], title="Scatter Plot")

        ax3 = fig.add_subplot(nrows, ncols, 3)
        create_histogram(ax3, data['hist'], title="Histogram")

        ax4 = fig.add_subplot(nrows, ncols, 4)
        create_bar_chart(ax4, data['bar'], title="Bar Chart")

        ax5 = fig.add_subplot(nrows, ncols, 5)
        create_box_plot(ax5, data['hist'], title="Box Plot")

        ax6 = fig.add_subplot(nrows, ncols, 6)
        create_heatmap(ax6, data['heatmap'], title="Heatmap")

        ax7 = fig.add_subplot(nrows, ncols, 7)
        create_error_bar(ax7, data['error'], title="Error Bar Plot")

        ax8 = fig.add_subplot(nrows, ncols, 8)
        create_contour_plot(ax8, data['contour'], title="Contour Plot")

        ax9 = fig.add_subplot(nrows, ncols, 9)
        create_math_plot(ax9, title="Mathematical Functions")

        ax10 = fig.add_subplot(nrows, ncols, 10)
        create_ml_plot(ax10, title="ROC Curves")

        fig.suptitle(
            f"Matplotlib {style_name.replace('_', ' ').title()} Style", fontsize=16, y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plt.savefig(f"{style_name}_demo.png", dpi=300)
        plt.savefig(f"{style_name}_demo.pdf")
        print(f"Saved {style_name}_demo.png and {style_name}_demo.pdf")

        plt.close(fig)


if __name__ == "__main__":
    test_styles()
