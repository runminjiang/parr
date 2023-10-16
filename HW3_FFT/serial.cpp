#include <iostream>
#include <complex>
#include <vector>
#include <cmath>

const double PI = 3.14159265358979323846f;

void fft(std::vector<std::complex<double>>& a, bool invert) {
    int n = a.size();
    if (n == 1)
        return;

    std::vector<std::complex<double>> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2*i];
        a1[i] = a[2*i+1];
    }
    fft(a0, invert);
    fft(a1, invert);

    double ang = 2 * PI / n * (invert ? -1 : 1);
    std::complex<double> w(1), wn(cos(ang), sin(ang));
    for (int i = 0; 2 * i < n; i++) {
        a[i] = a0[i] + w * a1[i];
        a[i+n/2] = a0[i] - w * a1[i];
        if (invert) {
            a[i] /= 2;
            a[i+n/2] /= 2;
        }
        w *= wn;
    }
}

int main() {
    int n;
    std::cout << "Enter number of points (should be a power of 2): ";
    std::cin >> n;

    std::vector<std::complex<double>> data(n);
    std::cout << "Enter " << n << " complex numbers (real and imaginary parts):\n";
    for(int i = 0; i < n; i++) {
        double realPart, imagPart;
        std::cin >> realPart >> imagPart;
        data[i] = std::complex<double>(realPart, imagPart);
    }

    fft(data, false);  // Forward FFT

    std::cout << "Transformed data:\n";
    for(const auto& val : data) {
        std::cout << val.real() << " + " << val.imag() << "i\n";
    }

    return 0;
}