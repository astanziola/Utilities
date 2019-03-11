# Wavelet transform using pytorch (no antitransform yet)

class wavelet_transformer:
    def __init__(self, wavelet_number=1, wavelet_type='db', device = 'cuda'):
        self.db_number = wavelet_number
        self.filters_high = []
        self.filters_low = []
        for wavelet_num in np.arange(wavelet_number)+1:
            w = pywt.Wavelet(wavelet_type + str(wavelet_num)) 
            dec_hi = torch.tensor( np.array(w.dec_hi), dtype=torch.float, device = device, requires_grad=False)
            self.filters_high.append(dec_hi.unsqueeze(0).unsqueeze(0))
            dec_lo = torch.tensor( np.array(w.dec_lo), dtype=torch.float, device = device, requires_grad=False)
            self.filters_low.append(dec_lo.unsqueeze(0).unsqueeze(0))
            
    def single_transform(self, signal=1, waveletnum = 0):
        # If one coefficient left, just return it
        if signal.shape[2] < self.filters_high[waveletnum].shape[2]:
            return signal
        # Else, return highpass and wavelet transformed low-pass, concatenated
        high_sig = torch.nn.functional.conv1d(signal, self.filters_high[waveletnum], stride=2)
        low_sig = torch.nn.functional.conv1d(signal, self.filters_low[waveletnum], stride=2)
        wt_low = self.single_transform(low_sig,waveletnum)
        conc_sig = torch.cat((high_sig, wt_low), 2)
        return conc_sig
        
    def transform(self, signal):
        signal = torch.reshape(signal,[1,1,-1])
        transf_signals = [self.single_transform(signal,wavenum) for wavenum in range(self.db_number)]
        final_signal = torch.cat(transf_signals, 2)/np.sqrt(self.db_number)
        return final_signal
