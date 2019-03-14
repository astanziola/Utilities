# Wavelet transform using pytorch (no antitransform yet). It works both for
# 1D and 2D data. For 2D, it assumes that a vector is given as input and
# automatically reshapes it into an image.

class wavelet_transformer:
	def __init__(self, wavelet_number=1, wavelet_type='db', device = 'cuda'):
		self.device = device
		self.db_number = wavelet_number
		self.filters_high = []
		self.filters_low = []
		self.filters_2D = []
		self.wavelet_type = wavelet_type	
		for wavelet_num in np.arange(wavelet_number)+1:
			w = pywt.Wavelet(wavelet_type + str(wavelet_num)) 
			dec_hi = torch.tensor( np.array(w.dec_hi), dtype=torch.float, device = self.device, requires_grad=False)
			self.filters_high.append(dec_hi.unsqueeze(0).unsqueeze(0))
			dec_lo = torch.tensor( np.array(w.dec_lo), dtype=torch.float, device = self.device, requires_grad=False)
			self.filters_low.append(dec_lo.unsqueeze(0).unsqueeze(0))
			# Construct 2d filters
			self.filters_2D.append(torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
					   dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
					   dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
					   dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0))

	def transform_2d(self, image, image_size=[256,256]):
		# Make the image from vector
		image = torch.reshape(image, [image_size[1],image_size[0]])
		# Pad image to nearest power of 2
		image = self.pad_image_to_power_of_2(image,resize=False)
		transf = [self.single_2d_transform(image, wn) for wn in range(self.db_number)]
		return torch.cat(transf)/np.sqrt(self.db_number)

	def single_2d_transform(self, image, filter_number = 0):
		filters = self.filters_2D[filter_number]
		return self.wt_2D(image, filters,wavenumber=filter_number)

	def wt_2D(self,vimg, filters, levels=1, wavenumber=0 ):
		h = vimg.size(2)
		w = vimg.size(3)
		if min([h,w]) <= 1:
			return vimg
			# res = vimg.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)
		padsize = wavenumber
		padded = torch.nn.functional.pad(vimg,(padsize,padsize,padsize,padsize))
		res = torch.nn.functional.conv2d(padded, torch.autograd.Variable(filters[:,None]).cuda(),stride=2)
		res[:,:1] = self.wt_2D(res[:,:1], filters, wavenumber=wavenumber)
		return res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)

	def pad_image_to_power_of_2(self, img, resize=False):#
		img = img.unsqueeze(0).unsqueeze(0)
		l,m,h,w = img.shape
		biggest = max([w,h])
		next_pow_2 = 2**np.ceil(np.log2(biggest)).astype(int)
		if resize:
			scaler = Resize((next_pow_2,next_pow_2))
			img = scaler(img)
		else:
			# Padding the image
			h_diff = next_pow_2 - h
			if h_diff > 0:
				img = torch.cat((img, torch.zeros([1,1,h_diff,w]).to(self.device)),2)
			w_diff = next_pow_2 - w
			if w_diff > 0:
				img = torch.cat((img, torch.zeros([1,1,next_pow_2,w_diff]).to(self.device)),3)   
		return img
			
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

	def test(self, N = 1000, M = 40):
		print('[ --- Test wavelet transform --- ]')
		print('  Device: ' + self.device)
		print('  Number of wavelets: ' + str(self.db_number	))
		print('  Wavelet family: ' + self.wavelet_type)
		print('[ ------------------------------ ]')

		s = np.random.randn(N)
		s = np.convolve(s, np.ones((M,))/M, mode='valid')
		plt.plot(s)
		plt.title('Sample original signal')
		plt.show()
		s = torch.from_numpy(s).float().to(self.device)
		s_hat = self.transform(s).cpu().numpy()
		s_hat = np.squeeze(s_hat)
		plt.plot(s_hat)
		plt.title('Transformed signal')
		plt.show()
