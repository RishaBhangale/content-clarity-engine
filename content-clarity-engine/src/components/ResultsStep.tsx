import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Download, FileSpreadsheet, FileQuestion } from "lucide-react";
import { Finding, UploadedFile } from "@/data/sampleData";
import FindingCard from "./FindingCard";
import DocumentMap from "./DocumentMap";
import { toast } from "sonner";

interface ResultsStepProps {
  files: UploadedFile[];
  findings: Finding[];
}

// ── Export Utilities ──────────────────────────────

function exportCSV(findings: Finding[], files: UploadedFile[]) {
  const headers = [
    "ID", "Severity", "Type", "Title", "Source A", "Source B",
    "Excerpt A", "Excerpt B", "Suggestion", "Confidence %",
  ];
  const escape = (s: string) => `"${(s || "").replace(/"/g, '""')}"`;

  const rows = findings.map((f) => [
    f.id, f.severity, f.type, escape(f.title),
    escape(f.sourceA), escape(f.sourceB || ""),
    escape(f.excerptA || ""), escape(f.excerptB || ""),
    escape(f.suggestion || ""), f.confidence,
  ].join(","));

  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `c1-findings-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  toast.success(`Exported ${findings.length} findings to CSV`);
}

function exportPDF(findings: Finding[], files: UploadedFile[]) {
  const critical = findings.filter((f) => f.severity === "critical").length;
  const warning = findings.filter((f) => f.severity === "warning").length;
  const info = findings.filter((f) => f.severity === "info").length;
  const date = new Date().toLocaleString();

  const severityColor: Record<string, string> = {
    critical: "#dc2626", warning: "#d97706", info: "#2563eb",
  };

  const findingsHTML = findings.map((f) => `
    <div style="border:1px solid #e5e7eb; border-left:4px solid ${severityColor[f.severity] || "#6b7280"}; border-radius:8px; padding:16px; margin-bottom:12px; page-break-inside:avoid;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <div>
          <span style="background:${severityColor[f.severity]}22; color:${severityColor[f.severity]}; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; text-transform:uppercase;">${f.severity}</span>
          <strong style="margin-left:10px; font-size:14px;">${f.title}</strong>
        </div>
        <span style="font-size:12px; color:#6b7280;">${f.confidence}% confidence</span>
      </div>
      <p style="font-size:12px; color:#6b7280; margin:4px 0 12px 0;">${f.sourceA} ↔ ${f.sourceB || "—"}</p>
      ${f.excerptA ? `<div style="background:#f9fafb; padding:10px; border-radius:6px; margin-bottom:8px; font-size:13px;"><strong>Document A:</strong> <em>"${f.excerptA}"</em></div>` : ""}
      ${f.excerptB ? `<div style="background:#f9fafb; padding:10px; border-radius:6px; margin-bottom:8px; font-size:13px;"><strong>Document B:</strong> <em>"${f.excerptB}"</em></div>` : ""}
      ${f.suggestion ? `<p style="font-size:13px; color:#4b5563;"><strong>Suggestion:</strong> ${f.suggestion}</p>` : ""}
    </div>
  `).join("");

  const html = `<!DOCTYPE html><html><head>
    <title>C1 Findings Report</title>
    <style>
      body { font-family: 'Inter', -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 24px; color: #1f2937; }
      h1 { font-size: 24px; margin: 0 0 4px 0; }
      .subtitle { color: #6b7280; font-size: 14px; margin-bottom: 24px; }
      .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 28px; }
      .metric { text-align: center; padding: 16px; border: 1px solid #e5e7eb; border-radius: 8px; }
      .metric .value { font-size: 28px; font-weight: 800; }
      .metric .label { font-size: 11px; color: #6b7280; margin-top: 2px; }
      .docs { font-size: 13px; color: #6b7280; margin-bottom: 20px; }
      h2 { font-size: 18px; margin: 24px 0 12px 0; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; }
      @media print { body { padding: 20px; } }
    </style>
  </head><body>
    <h1>C1 — Content Consistency Report</h1>
    <p class="subtitle">Generated on ${date}</p>
    <div class="metrics">
      <div class="metric"><div class="value" style="color:#4A6FA5;">${findings.length}</div><div class="label">Total Findings</div></div>
      <div class="metric"><div class="value" style="color:#dc2626;">${critical}</div><div class="label">Critical</div></div>
      <div class="metric"><div class="value" style="color:#d97706;">${warning}</div><div class="label">Warnings</div></div>
      <div class="metric"><div class="value" style="color:#2563eb;">${info}</div><div class="label">Info</div></div>
    </div>
    <p class="docs"><strong>Documents analyzed:</strong> ${files.map((f) => f.name).join(", ")}</p>
    <h2>Findings</h2>
    ${findingsHTML}
  </body></html>`;

  const printWindow = window.open("", "_blank");
  if (printWindow) {
    printWindow.document.write(html);
    printWindow.document.close();
    setTimeout(() => printWindow.print(), 500);
    toast.success("PDF report opened — use Ctrl/Cmd+P to save");
  } else {
    toast.error("Pop-up blocked — please allow pop-ups for this site");
  }
}

// ── Component ────────────────────────────────────

const ResultsStep = ({ files, findings }: ResultsStepProps) => {
  const critical = findings.filter((f) => f.severity === "critical");
  const warnings = findings.filter((f) => f.severity === "warning");
  const infos = findings.filter((f) => f.severity === "info");

  const contradictions = findings.filter((f) => f.type === "Contradiction");

  const METRICS = [
    { label: "Total Findings", value: findings.length, className: "text-primary" },
    { label: "Critical", value: critical.length, className: "text-destructive", bgClass: "bg-destructive-light" },
    { label: "Warnings", value: warnings.length, className: "text-warning-foreground", bgClass: "bg-warning-light" },
    { label: "Info", value: infos.length, className: "text-info", bgClass: "bg-info-light" },
  ];

  const renderFindings = (list: Finding[]) => {
    if (list.length === 0) {
      return (
        <div className="text-center py-12">
          <FileQuestion className="w-10 h-10 text-foreground-muted mx-auto mb-3" />
          <p className="text-foreground-muted text-sm">No findings in this category.</p>
        </div>
      );
    }
    return (
      <div className="space-y-3">
        {list.map((f) => (
          <FindingCard key={f.id} finding={f} />
        ))}
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4">
      {/* Export actions */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
        <h2 className="text-2xl font-bold">Results Dashboard</h2>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            className="gap-1.5"
            onClick={() => exportPDF(findings, files)}
          >
            <Download className="w-3.5 h-3.5" />
            Download PDF
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5"
            onClick={() => exportCSV(findings, files)}
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            Export CSV
          </Button>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {METRICS.map((m) => (
          <Card key={m.label} className={m.bgClass}>
            <CardContent className="pt-5 pb-4 text-center">
              <p className={`text-3xl font-extrabold ${m.className}`}>{m.value}</p>
              <p className="text-xs text-foreground-muted mt-1">{m.label}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Tabs + Document Map */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Tabs defaultValue="all">
            <TabsList className="mb-4">
              <TabsTrigger value="all">All Findings ({findings.length})</TabsTrigger>
              <TabsTrigger value="contradictions">Contradictions ({contradictions.length})</TabsTrigger>
            </TabsList>

            <ScrollArea className="h-[600px] pr-2">
              <TabsContent value="all" className="mt-0">{renderFindings(findings)}</TabsContent>
              <TabsContent value="contradictions" className="mt-0">{renderFindings(contradictions)}</TabsContent>
            </ScrollArea>
          </Tabs>
        </div>

        <div>
          <DocumentMap files={files} findings={findings} />
        </div>
      </div>
    </div>
  );
};

export default ResultsStep;
