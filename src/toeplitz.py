import numpy as np
from .fft import fft, ifft
from .utils import pad_to_power_of_two


def toeplitz_matvec(t_col: np.ndarray, t_row: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
        Mnoży macierz Toeplitza T (zdefiniowaną przez t_col i t_row) przez wektor x w O(n log n).
    
        t_col: pierwsza kolumna T (długość n)
        t_row: pierwszy wiersz T (długość n), t_row[0] == t_col[0]
        x:     wektor wejściowy (długość n)
        zwraca: y = T @ x
    """
    n = x.shape[0]
    assert t_col.shape[0] == n and t_row.shape[0] == n
    assert t_col[0] == t_row[0]

    x = pad_to_power_of_two(x)
    t_col = pad_to_power_of_two(t_col)
    t_row = pad_to_power_of_two(t_row)
    
    # Konstruowanie rozszerzonych wektorów długości 2n-1
    c = np.concatenate([t_row, np.array([t_col[0]]), np.flip(t_col[1:])])  # długość 2n
    x_ext = np.concatenate([x, np.zeros(len(x))])   # długość 2n

    counter = 0
    
    # FFT
    C, counter = fft(c, counter)
    X, counter = fft(x_ext, counter)
    print(f'fft counter {counter}')
    
    # Punktowy iloczyn i IFFT
    y_full = ifft(C * X)

    # Pobranie korelacyjnej części (indeksy n-1 do 2n-2)
    result = y_full[0:n].real

    return result