from fraud_project.utils import *


def calculate_psi(expected, actual, buckettype='quantiles', buckets=50, axis=0):
    def scale_range(inp, min_val, max_val):
        inp = inp - np.min(inp)
        inp = inp / (np.max(inp) / (max_val - min_val))
        return inp + min_val

    def psi_unit(exp_arr, act_arr, b):
        bp = np.arange(0, b + 1) / b * 100
        if buckettype == 'bins':
            bp = scale_range(bp, np.min(exp_arr), np.max(exp_arr))
        else:
            bp = np.percentile(exp_arr, bp)
        exp_pct = np.histogram(exp_arr, bp)[0] / len(exp_arr)
        act_pct = np.histogram(act_arr, bp)[0] / len(act_arr)
        def sub(e, a): return (max(e,1e-4) - max(a,1e-4)) * np.log(max(e,1e-4) / max(a,1e-4))
        return sum(sub(e,a) for e,a in zip(exp_pct, act_pct))

    if expected.ndim == 1:
        return np.array([psi_unit(expected, actual, buckets)])
    vals = np.zeros(expected.shape[1] if axis==0 else expected.shape[0])
    for i in range(len(vals)):
        arr_e = expected[:,i] if axis==0 else expected[i,:]
        arr_a = actual[:,i] if axis==0 else actual[i,:]
        vals[i] = psi_unit(arr_e, arr_a, buckets)
    return vals


def find_optimal_bins_with_knee(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    bin_range: Tuple[int,int]=(5,50),
    buckettype: str='quantiles'
) -> Tuple[Dict[str,int], Dict[str,pd.DataFrame]]:

    best_bins: Dict[str,int] = {}
    psi_tables: Dict[str,pd.DataFrame] = {}

    for col in expected_df.columns:
        exp = expected_df[col].dropna().values
        act = actual_df[col].dropna().values

        results = []
        for b in range(bin_range[0], bin_range[1]+1):
            psi_val = calculate_psi(exp, act, buckettype=buckettype, buckets=b)[0]
            results.append((b, psi_val))
        
        df = pd.DataFrame(results, columns=['bins','psi'])
        psi_tables[col] = df

        # Knee detection
        kl = KneeLocator(df['bins'], df['psi'], curve='concave', direction='decreasing')
        knee = kl.knee or df.loc[df['psi'].idxmax(), 'bins']
        best_bins[col] = int(knee)

    return best_bins, psi_tables