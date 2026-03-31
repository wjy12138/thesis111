import csv
import glob
import os

path = r'c:\Users\Don\Documents\毕设\thesis111\result\figure_csv\20260329_100521_lb96_lf1-2-3-4-5-6'
csvs = glob.glob(os.path.join(path, 'cross_site_metrics_lf*.csv'))

data = []
for f in csvs:
    with open(f, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)

print('\n=== LF=6 Metrics ===')
lf6 = [d for d in data if d['look_forward'] == '6']
for d in sorted(lf6, key=lambda x: (int(x['site_id']), x['model_name'])):
    print(f"Site {d['site_id']} {d['model_name']}: MSE={float(d['mse']):.3f}, MAE={float(d['mae']):.3f}, R2={float(d['r2']):.4f}")

models = ['CNN', 'CNN_LSTM', 'LSTM']
print('\n=== Average across sites for LF=6 ===')
for m in models:
    mses = [float(d['mse']) for d in lf6 if d['model_name'] == m]
    maes = [float(d['mae']) for d in lf6 if d['model_name'] == m]
    r2s = [float(d['r2']) for d in lf6 if d['model_name'] == m]
    print(f"{m} - Avg MSE: {sum(mses)/len(mses):.3f}, Avg MAE: {sum(maes)/len(maes):.3f}, Avg R2: {sum(r2s)/len(r2s):.4f}")

print('\n=== Trend over LF ===')
for lf in range(1, 7):
    lf_str = str(lf)
    for m in models:
        maes = [float(d['mae']) for d in data if d['model_name'] == m and d['look_forward'] == lf_str]
        r2s = [float(d['r2']) for d in data if d['model_name'] == m and d['look_forward'] == lf_str]
        mses = [float(d['mse']) for d in data if d['model_name'] == m and d['look_forward'] == lf_str]
        print(f"LF={lf} {m}: Avg MSE={sum(mses)/len(mses):.3f}, Avg MAE={sum(maes)/len(maes):.3f}, Avg R2={sum(r2s)/len(r2s):.4f}")
