# -*- coding: utf-8 -*-

"""
.. module:: statistics
   :synopsis: Defines an object with useful functions for calculating the 
			  morlet wavelet transforms of a 1-dimensional signal.

.. moduleauthor:: Ross McKinney
"""

import numpy as np

class Morlet:
	"""Object used to fit and transform signals via the Morlet wavelet transform.
	
	Parameters
	----------
	dt : int 
		Sampling frequency of signal to transform.
													
	freq : array-like | shape = [N]
		A 1d array of frequencies (Hz) to calculate wavelet across.
							
	omega0 : int (default = 5)
		Dimensionless wavelet parameter.

	Attributes
	----------
	n_freq_channels : int
		The number of frequencies to compute the wavelet transform over.
		Equivalent to freq.size.

	scales : np.ndarray | shape = [N] 
	"""
	def __init__(self, dt, freq, omega0 = 5):
		self.dt = dt
		self.freq = freq
		self.n_freq_channels = freq.size
		self.omega0 = omega0
		self.scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * freq)
		
		
	def transform(self, arr, return_wavlet_coeff = False):
		"""Transforms a signal into its characteristic
		frequency values using a morlet waveform.
		
		Parameters
		----------
		arr : np.ndarray | shape = [N] 
			Signal for which to compute Morlet wavelet transform.
								
		return_wavelet_coeff : boolean (default = False)
			Wheter or not to return the complex-valued wavelet coefficients 
			following Morlet transform.
		
		Returns
		-------
		amplitudes : np.ndarray | shape = [n_freq_channels, arr.size]
			Signal occupancy across different frequency bands.
								
		wavelet_coefficients : Not Implemented
			Complex-valued wavelet coefficients.
								
		References
		----------
		.. [1] Berman GJ et al., J R Soc Interface 2014.
		       https://github.com/gordonberman/MotionMapper/blob/master/
		       wavelet/fastWavelet_morlet_convolution_parallel.m
		"""
		
		n = arr.size
		pc_vals = arr
		amplitude = np.zeros((self.n_freq_channels, n))    
		
		if n % 2 == 1:
			pc_vals = np.hstack((arr, 0))
			n += 1
		
		#pad the ends of the pca_component array with zeros
		pc_vals = np.hstack((np.zeros(n/2), pc_vals, np.zeros(n/2))) 
		M = n
		n = pc_vals.size

		#generate indices to recover transformed pc_vals from within padding.
		ix = np.arange(M/2 + 1, M/2 + M, dtype = 'int') 
		
		omega = 2 * np.pi * np.arange(-n/2, n/2) / (n * self.dt)
		
		pc_ft = np.fft.fftshift(np.fft.fft(pc_vals))
		
		for i in range(self.n_freq_channels):
			m = self.wavelet(-omega * self.scales[i], self.omega0)
			q = np.fft.ifft(m * pc_ft) * np.sqrt(self.scales[i])
			q = q[ix]
			
			amplitude[i, :] = (np.abs(q) * np.pi**(-0.25) * 
				np.exp(0.25*(self.omega0 -np.sqrt(self.omega0**2 + 2))**2) / 
				np.sqrt(2 * self.scales[i]))
			
		return amplitude
		
	def wavelet(self, w, omega0):
		"""Generate a wavelet."""
		return np.pi**(-1/4) * np.exp(-0.5 * (w - omega0)**2);
		
		
		
		
