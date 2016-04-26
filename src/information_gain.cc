#include <ctgmath>

double information_gain (int n1, int n2, int p1, int p2)
{
    double n1p1 = n1 + p1;
    double n2p2 = n2 + p2;
    double n = n1 + n2;
    double p = p1 + p2;

    double pn = n + p;

    double current = 0.;
    double rem1 = 0.;
    double rem2 = 0.;

    double eps = 0.;

    if (pn > eps) {
        double q1 = n/pn;
        double q2 = p/pn;
        if (q1 > eps) {
            current += q1 * std::log2(q1);
        }
        if (q2 > eps) {
            current += q2 * std::log2(q2);
        }
        current *= -1.;
    } else {
        return 0.;
    }

    if (n1p1 > eps) {
        double q3 = n1/n1p1;
        double q4 = p1/n1p1;

        if (q3 > eps) {
            rem1 += q3 * std::log2(q3);
        }
        if (q4 > eps) {
            rem1 += q4 * std::log2(q4);
        }
        rem1 *= -n1p1/pn;
    }

    if (n2p2 > eps) {
        double q5 = n2/n2p2;
        double q6 = p2/n2p2;

        if (q5 > eps) {
            rem2 += q5 * std::log2(q5);
        }
        if (q6 > eps) {
            rem2 += q6 * std::log2(q6);
        }
        rem2 *= -n2p2/pn;
    }

    double remainder = rem1 + rem2;

    return current - remainder;
}
