// src/components/Polynomial.jsx
import React, { useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

// --- Utility: convert polynomial string to human-readable math ---
const formatPolynomial = (expr) => {
  if (!expr) return "";
  return expr
    .replace(/\*\*3/g, "³")
    .replace(/\*\*2/g, "²")
    .replace(/\*\*1/g, "")
    .replace(/\*/g, "")
    .replace(/-/g, "−")
    .replace(/\+/g, " + ")
    .replace(/  +/g, " ")
    .trim();
};

// --- Small UI helpers ---
const PrettyPre = ({ children }) => (
  <pre
    style={{
      whiteSpace: "pre-wrap",
      fontSize: "0.95rem",
      backgroundColor: "#f8f9fa",
      border: "1px solid #e6e9ef",
      borderRadius: 8,
      padding: "8px 12px",
      marginBottom: 12,
      display: "inline-block",
      width: "100%",
      overflowX: "auto",
    }}
  >
    {children}
  </pre>
);

// Inline array display
const InlineArray = ({ arr }) => {
  if (!arr) return <PrettyPre>None</PrettyPre>;
  if (!Array.isArray(arr)) return <PrettyPre>{String(arr)}</PrettyPre>;
  return (
    <PrettyPre>
      [
      {arr
        .map((x) =>
          Number.isFinite(x)
            ? Number(x).toFixed(6)
            : typeof x === "object"
            ? JSON.stringify(x)
            : String(x)
        )
        .join(", ")}
      ]
    </PrettyPre>
  );
};

const MatrixTable = ({ mat }) => {
  if (!Array.isArray(mat)) return <PrettyPre>{String(mat)}</PrettyPre>;
  return (
    <div
      className="table-responsive"
      style={{ maxHeight: "400px", overflowY: "auto", overflowX: "auto" }}
    >
      <table className="table table-sm table-bordered mb-0">
        <tbody>
          {mat.map((row, i) => (
            <tr key={i}>
              {row.map((v, j) => (
                <td
                  key={j}
                  style={{
                    minWidth: 80,
                    textAlign: "right",
                    fontFamily: "monospace",
                    padding: "4px 8px",
                    fontSize: "0.85rem",
                  }}
                >
                  {Number.isFinite(v) ? Number(v).toFixed(6) : String(v)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const AnswerCard = ({ label, title, children }) => (
  <div className="card mb-4 shadow-sm">
    <div className="card-body">
      <div className="d-flex align-items-start justify-content-between">
        <h6 className="mb-1">
          <span className="badge bg-primary me-2">{label}</span>
          <span className="fw-semibold">{title}</span>
        </h6>
      </div>
      <div className="mt-3">{children}</div>
    </div>
  </div>
);

// --- Main Component ---
const Polynomial = () => {
  const [n, setN] = useState(100);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e && e.preventDefault();
    setLoading(true);
    setError(null);
    setResp(null);
    try {
      const res = await fetch("http://localhost:5001/legendre_pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n: Number(n) }),
      });
      const data = await res.json();
      if (data.status === "success") {
        const result = data.result ?? data; // support both shapes
        setResp(result);
      } else setError("Server returned an error response");
    } catch (err) {
      console.error(err);
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container py-5">
      <h2 className="text-center mb-4 text-primary fw-bold">
        Modified Legendre Polynomial
      </h2>

      {/* Input Form */}
      <div className="card shadow p-3 mb-4">
        <form onSubmit={submit} className="row g-2 align-items-center">
          <div className="col-auto">
            <label className="form-label">Legendre order (n):</label>
            <input
              type="number"
              className="form-control mb-3"
              value={n}
              min={0}
              onChange={(e) => setN(e.target.value)}
            />
          </div>
          <hr/>
          <div className="col-auto">
            <button type="submit" className="btn btn-primary">
              Run Pipeline
            </button>
          </div>
          {loading && (
            <div className="col-auto">
              <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">Loading...</span>
              </div>
            </div>
          )}
        </form>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {/* Results */}
      {resp && (
        <>
          {/* A */}
          <AnswerCard
            label="A"
            title={`Modified (shifted) Legendre polynomial — P*_${n}(x)`}
          >
            <h6 className="text-muted small mb-2">Polynomial (pretty)</h6>
            <PrettyPre>
              {formatPolynomial(
                resp.P_shifted_pretty ??
                  resp.P_shifted_str ??
                  resp.polynomial
              )}
            </PrettyPre>

            {/* <h6 className="text-muted small mt-3">Polynomial (one-line)</h6> */}
            {/* <PrettyPre>
              {formatPolynomial(resp.P_shifted_str ?? resp.polynomial)}
            </PrettyPre> */}

            <h6 className="text-muted small mt-3">
              Coefficients (highest → lowest)
            </h6>
            <InlineArray arr={resp.coeffs_high ?? resp.coeffs_high} />
          </AnswerCard>

          {/* B */}
          <AnswerCard
            label="B"
            title="Companion matrix of the polynomial (Frobenius form)"
          >
            <h6 className="text-muted small mb-2">Companion Matrix (A)</h6>
            <MatrixTable mat={resp.companion_matrix ?? resp.companion_matrix} />
          </AnswerCard>

          {/* C */}
          <AnswerCard
            label="C"
            title="Roots = eigenvalues of companion (via LU decomposition)"
          >
            <h6 className="text-muted small mb-2">Eigenvalues (roots)</h6>
            <InlineArray arr={resp.eigenvalues ?? resp.eigenvalues} />

            <h6 className="text-muted small mt-3 mb-2">
              LU Decomposition (P, L, U)
            </h6>
            <div className="row">
              <div className="col-md-4">
                <h6 className="small text-muted">P</h6>
                <MatrixTable mat={resp.P_lu} />
              </div>
              <div className="col-md-4">
                <h6 className="small text-muted">L</h6>
                <MatrixTable mat={resp.L_lu} />
              </div>
              <div className="col-md-4">
                <h6 className="small text-muted">U</h6>
                <MatrixTable mat={resp.U_lu} />
              </div>
            </div>
          </AnswerCard>

          {/* D */}
          <AnswerCard label="D" title={`Solution of A·x = b (b = {1,2,..,n})`}>
            <h6 className="text-muted small mb-2">b vector</h6>
            <InlineArray
              arr={
                resp.b_vector ??
                Array.from({ length: Number(n) }, (_, i) => i + 1)
              }
            />

            <h6 className="text-muted small mt-3">Determinant det(A)</h6>
            <PrettyPre>{String(resp.determinant ?? resp.determinant)}</PrettyPre>

            <h6 className="text-muted small mt-3">Solution x (rounded)</h6>
            <InlineArray
              arr={resp.x_solution ?? resp.solution ?? resp.x_solution}
            />
          </AnswerCard>

          {/* E */}
          <AnswerCard
            label="E"
            title="Smallest & Largest roots (Newton–Raphson)"
          >
            <div className="row">
              <div className="col-md-6">
                <h6 className="small text-muted">Smallest (Newton)</h6>
                <PrettyPre>
                  {String(resp.newton_smallest ?? resp.newton_smallest)}
                </PrettyPre>
              </div>
              <div className="col-md-6">
                <h6 className="small text-muted">Largest (Newton)</h6>
                <PrettyPre>
                  {String(resp.newton_largest ?? resp.newton_largest)}
                </PrettyPre>
              </div>
            </div>

            <h6 className="text-muted small mt-3">Newton iteration logs</h6>
            <PrettyPre>
              {(resp.newton_logs ?? resp.logs ?? []).join("\n")}
            </PrettyPre>
          </AnswerCard>

          {/* Summary */}
          <div className="card p-3 mb-5">
            <h6 className="fw-semibold">Summary</h6>
            <ul className="mb-0">
              <li>
                Polynomial degree:{" "}
                {resp.coeffs_high?.length
                  ? resp.coeffs_high.length - 1
                  : "?"}
              </li>
              <li>
                Number of eigenvalues returned:{" "}
                {resp.eigenvalues?.length ?? "?"}
              </li>
              <li>Determinant: {String(resp.determinant ?? resp.determinant)}</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default Polynomial;
