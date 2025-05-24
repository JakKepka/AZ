import numpy as np

def fft(x: np.ndarray, counter) -> np.ndarray:
    """
    Oblicza dyskretną transformatę Fouriera (FFT) wektora x.
    Założenie: len(x) jest potęgą dwójki.
    """
    n = x.shape[0]
    counter += 1

    if n == 1:
        return x.copy(), counter
    even, counter = fft(x[::2], counter)
    odd, counter = fft(x[1::2], counter)

    factor = np.exp(-2j * np.pi * np.arange(n) / n)

    return np.concatenate([even + factor[:n//2] * odd,
                           even - factor[:n//2] * odd]), counter

def ifft(X: np.ndarray) -> np.ndarray:
    """
    Oblicza odwrotną FFT (IFFT) wektora X.
    """
    n = X.shape[0]
    x_conj = np.conjugate(X)

    counter = 0 
    y, counter = fft(x_conj, counter)
    print(f'ifft counter {counter}')
    return np.conjugate(y) / n