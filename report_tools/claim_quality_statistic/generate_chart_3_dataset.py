import matplotlib.pyplot as plt
import pandas as pd
import os


def generate_combined_chart():
    # 1. Danh sách các method (Trục Y)
    methods = [
        'FActScore (ministral-3:14b)', 'FActScore (phi4:14b)', 'FActScore (deepseek-r1:14b)',
        'MedScore (ministral-3:14b)', 'MedScore (phi4:14b)', 'MedScore (deepseek-r1:14b)',
        'MedScore (GPT-4o-mini)',
        'MedScore-LMS (ministral-3:14b)', 'MedScore-LMS (phi4:14b)', 'MedScore-LMS (deepseek-r1:14b)'
    ]

    # 2. Cấu hình dữ liệu cho cả 3 dataset
    datasets_data = {
        "AskDocsAI": {
            "totals": [4978, 3925, 2848, 1516, 2532, 860, 1209, 1204, 1288, 1027],
            "counts_data": {
                'Valid':                  [1463, 1258, 731,  950, 1291, 160, 930, 874, 892, 763],
                'Unverifiable':           [273,  274,  197,  0,   171,  82,  0,   0,   4,   4],
                'Hallucinated':           [311,  170,  113,  403, 230,  24,  124, 128, 50,  37],
                'Incomplete':             [728,  600,  496,  98,  365,  467, 73,  46,  50,  49],
                'Incorrectly-structured': [349,  378,  223,  0,   213,  56,  4,   7,   40,  36],
                'Context-dependent':      [1398, 1126, 1055, 17,  204,  64,  72,  28,  87,  86],
                'Redundant':              [452,  117,  31,   48,  58,   7,   6,   121, 165, 52]
            }
        },
        "MedLFQA": {
            "totals": [4405,2897,2294,1525,1996,596,0,1149,1167,980],
            "counts_data": {
                'Valid':                  [1294,1153,753,1080,1165,104,0,968,946,822],
                'Unverifiable':           [152,98,62,10,71,48,0,1,0,6],
                'Hallucinated':           [471,196,155,187,139,12,0,40,29,27],
                'Incomplete':             [892,560,431,175,259,365,0,45,46,49],
                'Incorrectly-structured': [361,299,113,8,166,34,0,3,10,8],
                'Context-dependent':      [853,496,729,31,157,32,0,7,24,26],
                'Redundant':              [382,95,51,34,39,1,0,85,112,42]
            }
        },
        "ChatDoctor-iCliniq": {
            "totals": [6810,4044,2945,1356,2733,841,0,987,1092,932],
            "counts_data": {
                'Valid':                  [1650,1220,762,749,1068,111,0,764,789,690],
                'Unverifiable':           [665,317,284,10,350,139,0,2,8,24],
                'Hallucinated':           [707,221,252,402,285,27,0,81,72,52],
                'Incomplete':             [1207,783,478,126,408,452,0,39,39,50],
                'Incorrectly-structured': [615,381,230,6,343,70,0,4,14,6],
                'Context-dependent':      [1501,988,881,17,240,39,0,26,72,68],
                'Redundant':              [465,134,58,46,39,3,0,71,98,42]
            }
        }
    }

    colors = ['#27ae60', '#e74c3c', '#f39c12', '#3498db', '#9b59b6', '#d35400', '#7f8c8d']

    # 3. Bố cục dọc: 3 hàng x 1 cột, dùng chung trục X.
    #    Kích thước ~ bề rộng vùng chữ A4 (15.5 cm ≈ 6.1 in) để chèn vào Word
    #    ở 100% mà không bị thu nhỏ -> cỡ chữ trong hình giữ đúng như khai báo.
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.3, 9.6), sharex=True)

    for i, (dataset_name, data) in enumerate(datasets_data.items()):
        ax = axes[i]
        totals = data['totals']
        counts_data = data['counts_data']

        df_counts = pd.DataFrame(counts_data, index=methods)
        safe_totals = [t if t > 0 else 1 for t in totals]
        df_pct = df_counts.div(safe_totals, axis=0) * 100

        df_pct.plot(kind='barh', stacked=True, color=colors, ax=ax, width=0.8,
                    edgecolor='white', linewidth=0.5)

        # 4. Nhãn % bên trong thanh (chỉ hiện khi > 5% để không chồng chữ)
        for j, col in enumerate(df_counts.columns):
            for k, method in enumerate(methods):
                patch = ax.patches[j * len(methods) + k]
                width = patch.get_width()
                if width > 5.0:
                    x_pos = patch.get_x() + width / 2
                    y_pos = patch.get_y() + patch.get_height() / 2
                    ax.text(x_pos, y_pos, f"{width:.0f}%", ha='center', va='center',
                            fontsize=7, color='white', fontweight='bold')

        # Nhãn tổng số bên phải; xử lý GPT-4o-mini không có dữ liệu
        for k, total in enumerate(totals):
            if methods[k] == 'MedScore (GPT-4o-mini)' and total == 0:
                ax.text(101.5, k, "—", va='center', fontsize=8, color='#333333', fontweight='bold')
                ax.text(50, k, "— Not evaluated on this dataset —",
                        ha='center', va='center', fontsize=8, color='gray', fontstyle='italic')
            else:
                ax.text(101.5, k, f"N={total}", va='center', fontsize=7.5, color='#333333')

        ax.set_title(dataset_name, fontsize=10, fontweight='bold', pad=4)
        ax.set_xlim(0, 115)
        ax.set_xticks(range(0, 101, 20))
        ax.invert_yaxis()           # mỗi subplot có trục Y riêng nên lật từng cái
        ax.set_ylabel('')
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.get_legend().remove()

    axes[-1].set_xlabel('Percentage (%)', fontsize=9)

    # 5. Legend chung ở dưới cùng
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.2)
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, frameon=False, fontsize=8, handlelength=1.2, columnspacing=1.0)

    plt.savefig('combined_claim_quality_statistic_vertical.pdf', bbox_inches='tight')
    plt.savefig('combined_claim_quality_statistic_vertical.png', dpi=300, bbox_inches='tight')
    print(f"Files were created successfully in {os.getcwd()}")


if __name__ == "__main__":
    generate_combined_chart()