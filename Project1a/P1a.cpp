#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <cctype>

using namespace std;
namespace fs = std::filesystem;

const double R = 8.314462618;


// User Menu
string lower(string s) {
    for (char &c : s) c = tolower((unsigned char)c);
    return s;}

double volume_factor(string unit) {
    unit = lower(unit);
    if (unit == "m^3/mol") return 1.0;
    if (unit == "dm^3/mol" || unit == "l/mol") return 1e-3;
    if (unit == "cm^3/mol") return 1e-6;
    throw runtime_error("Unknown volume unit: " + unit);}

double pressure_si(double p, string unit) {
    unit = lower(unit);
    if (unit == "pa") return p;
    if (unit == "kpa") return p * 1e3;
    if (unit == "mpa") return p * 1e6;
    if (unit == "bar") return p * 1e5;
    if (unit == "kilobar" || unit == "kbar") return p * 1e8;
    if (unit == "atm") return p * 101325.0;
    if (unit == "torr") return p * 101325.0 / 760.0;
    if (unit == "mmhg") return p * 133.322387415;
    throw runtime_error("Unknown pressure unit: " + unit);}


// Virial EOS
double virial(double V, double T, const vector<double>& B) {
    double sum = 1.0;
    for (int i = 0; i < (int)B.size(); i++) sum += B[i] / pow(V, i + 1);
    return R * T / V * sum;}

vector<double> residuals(const vector<double>& V, const vector<double>& P, double T, const vector<double>& B) {
    vector<double> r(V.size());
    for (int i = 0; i < (int)V.size(); i++) r[i] = P[i] - virial(V[i], T, B);
    return r;}

double sse(const vector<double>& V, const vector<double>& P, double T, const vector<double>& B) {
    double S = 0.0;
    for (int i = 0; i < (int)V.size(); i++) {
        double r = P[i] - virial(V[i], T, B);
        S += r * r;}
    return S;}


// Jacobian
vector<vector<double>> jacobian(const vector<double>& V, double T, int M) {
    vector<vector<double>> J(V.size(), vector<double>(M));
    for (int i = 0; i < (int)V.size(); i++)
        for (int j = 0; j < M; j++)
            J[i][j] = R * T / pow(V[i], j + 2);
    return J;}

void normal_equations(const vector<vector<double>>& J, const vector<double>& r,
                      vector<vector<double>>& alpha, vector<double>& beta) {
    int n = J.size(), m = J[0].size();
    alpha.assign(m, vector<double>(m, 0.0));
    beta.assign(m, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            beta[j] += J[i][j] * r[i];
            for (int k = 0; k < m; k++) alpha[j][k] += J[i][j] * J[i][k];
        }}}



// Gaussian elimination
bool solve(vector<vector<double>> A, vector<double> b, vector<double>& x) {
    int n = A.size();

    for (int i = 0; i < n; i++) {
        int pivot = i;
        for (int k = i + 1; k < n; k++)
            if (fabs(A[k][i]) > fabs(A[pivot][i])) pivot = k;

        if (!isfinite(A[pivot][i]) || fabs(A[pivot][i]) < 1e-30) return false;

        swap(A[i], A[pivot]);
        swap(b[i], b[pivot]);

        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j < n; j++) A[k][j] -= factor * A[i][j];
            b[k] -= factor * b[i];
        }}

    x.assign(n, 0.0);

    for (int i = n - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < n; j++) sum -= A[i][j] * x[j];
        if (!isfinite(A[i][i]) || fabs(A[i][i]) < 1e-30) return false;
        x[i] = sum / A[i][i];
        if (!isfinite(x[i])) return false;}

    return true;}

bool inverse(const vector<vector<double>>& A, vector<vector<double>>& inv) {
    int n = A.size();
    inv.assign(n, vector<double>(n));

    for (int j = 0; j < n; j++) {
        vector<double> e(n, 0.0), x;
        e[j] = 1.0;
        if (!solve(A, e, x)) return false;
        for (int i = 0; i < n; i++) inv[i][j] = x[i];}

    return true;}



// Levenberg-Marquardt
struct LMResult {
    vector<double> B;
    double S;};

LMResult levenberg_marquardt(const vector<double>& V, const vector<double>& P, double T,
                             vector<double> B, const string& filename) {
    double lambda = 1e-3;
    double S = sse(V, P, T, B);
    int small = 0;

    fs::create_directories("result");
    string output = "result/" + fs::path(filename).stem().string() + "_LM_cycles.csv";
    ofstream log(output);

    log << "Cycle,Lambda,SSE";
    for (int i = 0; i < (int)B.size(); i++) log << ",B" << i + 2;
    log << ",Status\n";
    log << scientific << setprecision(12);

    for (int cycle = 1; cycle <= 1000; cycle++) {
        vector<double> r = residuals(V, P, T, B), beta, delta;
        vector<vector<double>> J = jacobian(V, T, B.size()), alpha;
        normal_equations(J, r, alpha, beta);

        double used_lambda = lambda;
        vector<vector<double>> modified = alpha;
        for (int i = 0; i < (int)B.size(); i++) modified[i][i] *= 1.0 + lambda;

        string status = "REJECT";
        bool solved = solve(modified, beta, delta);

        if (solved) {
            vector<double> trial = B;
            for (int i = 0; i < (int)B.size(); i++) trial[i] += delta[i];

            double newS = sse(V, P, T, trial);

            if (isfinite(newS) && newS < S) {
                double decrease = S - newS;
                double fractional = decrease / max(fabs(S), 1e-30);

                B = trial;
                S = newS;
                lambda /= 10.0;
                status = "ACCEPT";

                if (fractional < 1e-5) small++;
                else small = 0;
            } else {
                lambda *= 10.0;
            }
        } else {
            lambda *= 10.0;
        }

        log << cycle << "," << used_lambda << "," << S;
        for (double b : B) log << "," << b;
        log << "," << status << "\n";

        if (small >= 3) break;
        if (!isfinite(lambda) || lambda > 1e100) break;}

    log.close();
    return {B, S};
}



// Statistics
struct Statistics {
    int df;
    double variance, rmse, r2, adjusted_r2, rfactor;
    vector<double> se;
    vector<vector<double>> covariance, correlation;};

Statistics calculate_statistics(const vector<double>& V, const vector<double>& P,
                                double T, const vector<double>& B) {
    int N = P.size(), M = B.size(), df = N - M;
    if (df <= 0) throw runtime_error("Not enough data points.");

    vector<double> r = residuals(V, P, T, B);
    double S = 0.0, mean = 0.0, total = 0.0, absP = 0.0, absR = 0.0;

    for (double x : r) S += x * x;
    for (double x : P) mean += x;
    mean /= N;

    for (int i = 0; i < N; i++) {
        total += pow(P[i] - mean, 2);
        absP += fabs(P[i]);
        absR += fabs(r[i]);}

    double variance = S / df;
    double rmse = sqrt(S / N);
    double r2 = total == 0.0 ? NAN : 1.0 - S / total;
    double adjusted = total == 0.0 ? NAN : 1.0 - (S / df) / (total / (N - 1));
    double rfactor = absP == 0.0 ? NAN : absR / absP;

    vector<vector<double>> J = jacobian(V, T, M), alpha, C;
    vector<double> beta;
    normal_equations(J, r, alpha, beta);

    if (!inverse(alpha, C)) throw runtime_error("Cannot calculate covariance matrix.");

    vector<vector<double>> covariance(M, vector<double>(M));
    vector<vector<double>> correlation(M, vector<double>(M));
    vector<double> se(M);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++)
            covariance[i][j] = variance * C[i][j];

    for (int i = 0; i < M; i++)
        se[i] = covariance[i][i] >= 0.0 ? sqrt(covariance[i][i]) : NAN;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++)
            correlation[i][j] = (se[i] != 0.0 && se[j] != 0.0)
                ? covariance[i][j] / (se[i] * se[j]) : NAN;

    return {df, variance, rmse, r2, adjusted, rfactor, se, covariance, correlation};
}



// Output statistics
void print_statistics(const vector<double>& B, double S, const Statistics& s) {
    cout << scientific << setprecision(8);

    cout << "\nOptimized coefficients:\n";
    for (int i = 0; i < (int)B.size(); i++)
        cout << "B" << i + 2 << " = " << B[i] << "   SE = " << s.se[i] << "\n";

    cout << "\nSSE = " << S
         << "\nDegrees of freedom = " << s.df
         << "\nSample variance = " << s.variance
         << "\nRMSE = " << s.rmse
         << "\nR^2 = " << s.r2
         << "\nAdjusted R^2 = " << s.adjusted_r2
         << "\nR-factor = " << s.rfactor << "\n";

    cout << "\nCovariance matrix:\n";
    for (const auto& row : s.covariance) {
        for (double x : row) cout << setw(15) << x << " ";
        cout << "\n";
    }

    cout << "\nCorrelation matrix:\n";
    for (const auto& row : s.correlation) {
        for (double x : row) cout << setw(15) << x << " ";
        cout << "\n";
    }
}


// Save CSV
void save_fit_csv(const string& name, const vector<double>& V, const vector<double>& P,
                  double T, const vector<double>& B) {
    fs::create_directories("result");
    string output = "result/" + fs::path(name).stem().string() + "_fit.csv";

    ofstream f(output);
    f << "Volume_m3_per_mol,Experimental_Pa,Fitted_Pa,Residual_Pa\n";
    f << scientific << setprecision(12);

    for (int i = 0; i < (int)V.size(); i++) {
        double fitted = virial(V[i], T, B);
        f << V[i] << "," << P[i] << "," << fitted << "," << P[i] - fitted << "\n";
    }
}


// SVG plot
void create_plot(const string& name, const vector<double>& V, const vector<double>& P,
                 double T, const vector<double>& B) {
    fs::create_directories("result");
    string output = "result/" + fs::path(name).stem().string() + "_fit.svg";

    double xmin = *min_element(V.begin(), V.end());
    double xmax = *max_element(V.begin(), V.end());
    double ymin = *min_element(P.begin(), P.end());
    double ymax = *max_element(P.begin(), P.end());

    vector<pair<double,double>> curve;

    for (int i = 0; i < 500; i++) {
        double x = xmin + (xmax - xmin) * i / 499.0;
        double y = virial(x, T, B);
        curve.push_back({x, y});
        ymin = min(ymin, y);
        ymax = max(ymax, y);
    }

    double dx = xmax - xmin;
    double dy = ymax - ymin;
    if (dx == 0.0) dx = 1.0;
    if (dy == 0.0) dy = 1.0;

    xmin -= 0.05 * dx; xmax += 0.05 * dx;
    ymin -= 0.05 * dy; ymax += 0.05 * dy;

    const int W = 900, H = 600, L = 90, Right = 40, Top = 40, Bottom = 70;

    auto X = [&](double x) {
        return L + (x - xmin) / (xmax - xmin) * (W - L - Right);
    };

    auto Y = [&](double y) {
        return H - Bottom - (y - ymin) / (ymax - ymin) * (H - Top - Bottom);
    };

    ofstream f(output);

    f << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"600\">";
    f << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>";
    f << "<text x=\"450\" y=\"25\" text-anchor=\"middle\" font-size=\"20\">Virial EOS Fit</text>";
    f << "<line x1=\"" << L << "\" y1=\"" << H-Bottom << "\" x2=\"" << W-Right << "\" y2=\"" << H-Bottom << "\" stroke=\"black\"/>";
    f << "<line x1=\"" << L << "\" y1=\"" << Top << "\" x2=\"" << L << "\" y2=\"" << H-Bottom << "\" stroke=\"black\"/>";
    f << "<text x=\"450\" y=\"585\" text-anchor=\"middle\">Molar Volume (m^3/mol)</text>";
    f << "<text x=\"20\" y=\"300\" text-anchor=\"middle\" transform=\"rotate(-90 20 300)\">Pressure (Pa)</text>";

    int skip = max(1, (int)V.size() / 1500);
    for (int i = 0; i < (int)V.size(); i += skip)
        f << "<circle cx=\"" << X(V[i]) << "\" cy=\"" << Y(P[i]) << "\" r=\"2\" fill=\"black\"/>";

    f << "<polyline fill=\"none\" stroke=\"red\" stroke-width=\"2\" points=\"";
    for (auto &p : curve) f << X(p.first) << "," << Y(p.second) << " ";
    f << "\"/></svg>";
}







// Main
int main() {
    try {
        cout << string(50, '-') << "\n";
        cout << "Project 1a - \n";
        cout << string(50, '-') << "\n";
        vector<string> filenames;

        if (!fs::exists("file")) throw runtime_error("Folder 'file' not found.");

        for (const auto& entry : fs::directory_iterator("file")) {
            if (entry.is_regular_file() && lower(entry.path().extension().string()) == ".txt")
                filenames.push_back(entry.path().filename().string());
        }

        sort(filenames.begin(), filenames.end());
        if (filenames.empty()) throw runtime_error("No .txt files found.");

        for (int i = 0; i < (int)filenames.size(); i++)
            cout << "[" << i + 1 << "] " << filenames[i] << "\n";

        cout << "Select file: ";
        int choice;
        cin >> choice;

        if (choice < 1 || choice > (int)filenames.size())
            throw runtime_error("Invalid selection.");

        string name = filenames[choice - 1];
        ifstream f(fs::path("file") / name);

        if (!f) throw runtime_error("Cannot open file.");

        string comment;
        getline(f >> ws, comment);
        cout << comment << "\n";

        string keyword;
        int M;
        f >> keyword >> M;

        if (lower(keyword) != "virial" || M < 1 || M > 10)
            throw runtime_error("Invalid virial specification.");

        vector<double> B(M);
        for (double& x : B) f >> x;

        string temp_word, temp_unit, volume_unit, pressure_unit;
        double T;

        f >> temp_word >> T >> temp_unit;
        f >> volume_unit >> pressure_unit;

        if (lower(temp_word) != "temp")
            throw runtime_error("Invalid temperature line.");

        double vf = volume_factor(volume_unit);

        // B2 has volume^1 units, B3 volume^2, etc.
        for (int i = 0; i < M; i++) B[i] *= pow(vf, i + 1);

        vector<double> V, P;
        double v, p;

        while (f >> v >> p) {
            V.push_back(v * vf);
            P.push_back(pressure_si(p, pressure_unit));
        }

        if (V.size() <= B.size())
            throw runtime_error("Not enough data points.");

        LMResult fit = levenberg_marquardt(V, P, T, B, name);
        Statistics stats = calculate_statistics(V, P, T, fit.B);

        print_statistics(fit.B, fit.S, stats);
        save_fit_csv(name, V, P, T, fit.B);
        create_plot(name, V, P, T, fit.B);
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}