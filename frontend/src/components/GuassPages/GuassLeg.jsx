// src/components/GaussLeg.jsx
import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend);

// -----------------------------------------------------------
// Utility pretty printer (inline arrays)
// -----------------------------------------------------------
const PrettyPre = ({ children }) => (
  <pre
    style={{
      whiteSpace: "pre-wrap",
      backgroundColor: "#f8f9fa",
      border: "1px solid #e6e9ef",
      borderRadius: 8,
      padding: 10,
      fontSize: "0.9rem",
      marginBottom: 10,
      overflowX: "auto",
    }}
  >
    {children}
  </pre>
);

const InlineArray = ({ arr }) => {
  if (!arr) return <PrettyPre>None</PrettyPre>;
  if (!Array.isArray(arr)) return <PrettyPre>{String(arr)}</PrettyPre>;
  return (
    <PrettyPre>
      [{arr.map((x) => (Number.isFinite(x) ? Number(x).toFixed(6) : String(x))).join(", ")}]
    </PrettyPre>
  );
};

const MatrixTable = ({ title, mat }) => (
  <div className="mb-4">
    <h6 className="fw-semibold text-secondary">{title}</h6>
    <div className="table-responsive">
      <table className="table table-sm table-bordered">
        <tbody>
          {Array.isArray(mat) &&
            mat.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td
                    key={j}
                    style={{
                      minWidth: 70,
                      textAlign: "right",
                      fontFamily: "monospace",
                      fontSize: "0.8rem",
                      padding: "4px 8px",
                    }}
                  >
                    {Number.isFinite(v) ? v.toFixed(5) : String(v)}
                  </td>
                ))}
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  </div>
);

const GaussLeg = () => {
  const [n, setN] = useState(4);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState(null);

  const runPipeline = async (e) => {
    e && e.preventDefault();
    setLoading(true);
    setError(null);
    setResp(null);
    try {
      const res = await fetch("http://localhost:5001/gauss_legendre", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n: Number(n) }),
      });
      const data = await res.json();
      if (data.status === "success") setResp(data);
      else setError(data.message || "Server error");
    } catch (err) {
      console.error(err);
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  // Chart data
  const chartData = resp
    ? {
        labels: resp.roots_gw.map((x) => x.toFixed(4)),
        datasets: [
          {
            label: "Golub–Welsch Weights",
            data: resp.weights_gw,
            borderColor: "rgba(0,123,255,0.8)",
            backgroundColor: "rgba(0,123,255,0.5)",
            tension: 0.4,
            pointRadius: 6,
            fill: false,
          },
          {
            label: "Lagrange Weights",
            data: resp.weights_lag,
            borderColor: "rgba(220,53,69,0.8)",
            backgroundColor: "rgba(220,53,69,0.5)",
            tension: 0.4,
            pointStyle: "triangle",
            pointRadius: 6,
            fill: false,
          },
        ],
      }
    : null;

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: "top" },
      tooltip: { mode: "nearest" },
    },
    scales: {
      x: {
        title: { display: true, text: "Roots (ξ)" },
      },
      y: {
        title: { display: true, text: "Weights (w)" },
      },
    },
  };

  return (
    <div className="container-fluid py-5 d-flex justify-content-center">
      <div style={{ width: "80vw", maxWidth: "1600px" }}>
        <h2 className="text-center text-primary fw-bold mb-4">
          Gauss–Legendre Collocation & Integration
        </h2>

        {/* Input Form */}
        <div className="card shadow p-4 mb-4 border-0 rounded-4">
          <form onSubmit={runPipeline} className="row g-3 align-items-center">
            <div className="col-md-4">
              <label className="form-label fw-semibold">Order (n):</label>
              <input
                type="number"
                className="form-control"
                value={n}
                min={2}
                max={64}
                onChange={(e) => setN(e.target.value)}
              />
            </div>
            <hr/>
            <div className="col-md-4">
              <button
                className="btn btn-primary w-100"
                type="submit"
                disabled={loading}
              >
                {loading ? "Running..." : "Run Analysis"}
              </button>
            </div>
          </form>

          {!resp && !loading && (
            <div className="mt-4">
              <p className="text-secondary mb-2 fw-semibold">
                This pipeline performs:
              </p>
              <ul className="list-group small">
                <li className="list-group-item">
                  A. Modified Legendre polynomial coefficients.
                </li>
                <li className="list-group-item">
                  B. Golub–Welsch and Lagrange nodes & weights.
                </li>
                <li className="list-group-item">
                  C. Collocation derivative matrices A₁ and B.
                </li>
                <li className="list-group-item">
                  D. Plot of Weights vs Roots (both methods).
                </li>
              </ul>
            </div>
          )}
        </div>

        {error && <div className="alert alert-danger">{error}</div>}

        {/* Output */}
        {resp && (
          <div className="card shadow p-4 border-0 rounded-4">
            <h5 className="fw-bold text-dark mb-3">Results for n = {resp.n}</h5>

            <h6 className="text-muted small mb-2">
              Polynomial Coefficients (highest → lowest)
            </h6>
            <InlineArray arr={resp.coeffs} />

            <h6 className="text-muted mt-3 mb-2">
              Roots and Weights (Golub–Welsch)
            </h6>
            <PrettyPre>
              {resp.roots_gw
                .map(
                  (r, i) =>
                    `Root[${i + 1}] = ${r.toFixed(6)},  W = ${resp.weights_gw[
                      i
                    ].toFixed(6)}`
                )
                .join("\n")}
            </PrettyPre>

            <h6 className="text-muted mt-3 mb-2">
              Weights (Lagrange Integration)
            </h6>
            <PrettyPre>
              {resp.weights_lag
                .map(
                  (w, i) =>
                    `Root[${i + 1}] = ${resp.roots_gw[i].toFixed(
                      6
                    )},  W = ${w.toFixed(6)}`
                )
                .join("\n")}
            </PrettyPre>

            <MatrixTable title="Matrix A₁ (y')" mat={resp.A1} />
            <MatrixTable title="Matrix B (y'')" mat={resp.B} />

            <h6 className="fw-semibold mt-3 mb-2">Collocation Points (x)</h6>
            <InlineArray arr={resp.x_nodes} />

            <hr />
            <h5 className="fw-bold text-dark mb-3">Weights vs Roots Plot</h5>
            <div
              style={{
                background: "#fff",
                padding: 25,
                borderRadius: 12,
                width: "100%",
                height: "500px",
              }}
            >
              <Line
                key={JSON.stringify(resp.roots_gw)}
                data={chartData}
                options={chartOptions}
              />
            </div>

            <hr />
            <h6 className="fw-semibold mt-4">Logs</h6>
            <PrettyPre>{resp.logs.join("\n")}</PrettyPre>
          </div>
        )}
      </div>
    </div>
  );
};

export default GaussLeg;
