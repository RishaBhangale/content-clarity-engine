import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Download, FileSpreadsheet, Share2 } from "lucide-react";
import { Finding, UploadedFile, SAMPLE_FINDINGS } from "@/data/sampleData";
import FindingCard from "./FindingCard";
import DocumentMap from "./DocumentMap";
import { toast } from "sonner";

interface ResultsStepProps {
  files: UploadedFile[];
}

const ResultsStep = ({ files }: ResultsStepProps) => {
  const findings = SAMPLE_FINDINGS;
  const critical = findings.filter((f) => f.severity === "critical");
  const warnings = findings.filter((f) => f.severity === "warning");
  const infos = findings.filter((f) => f.severity === "info");

  const contradictions = findings.filter((f) => f.type === "Contradiction");
  const drift = findings.filter((f) => f.type === "Semantic Drift");
  const staleAndTerm = findings.filter((f) => f.type === "Stale Reference" || f.type === "Terminology");

  const METRICS = [
    { label: "Total Findings", value: findings.length, className: "text-primary" },
    { label: "Critical", value: critical.length, className: "text-destructive", bgClass: "bg-destructive-light" },
    { label: "Warnings", value: warnings.length, className: "text-warning-foreground", bgClass: "bg-warning-light" },
    { label: "Info", value: infos.length, className: "text-info", bgClass: "bg-info-light" },
  ];

  const renderFindings = (list: Finding[]) => (
    <div className="space-y-3">
      {list.map((f) => (
        <FindingCard key={f.id} finding={f} />
      ))}
    </div>
  );

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
            onClick={() => toast.success("Report downloaded")}
          >
            <Download className="w-3.5 h-3.5" />
            Download PDF
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5"
            onClick={() => toast.success("CSV exported")}
          >
            <FileSpreadsheet className="w-3.5 h-3.5" />
            Export CSV
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5"
            onClick={() => toast.success("Share link copied")}
          >
            <Share2 className="w-3.5 h-3.5" />
            Share
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
              <TabsTrigger value="drift">Semantic Drift ({drift.length})</TabsTrigger>
              <TabsTrigger value="stale">Stale Refs & Terminology ({staleAndTerm.length})</TabsTrigger>
            </TabsList>

            <ScrollArea className="h-[600px] pr-2">
              <TabsContent value="all" className="mt-0">{renderFindings(findings)}</TabsContent>
              <TabsContent value="contradictions" className="mt-0">{renderFindings(contradictions)}</TabsContent>
              <TabsContent value="drift" className="mt-0">{renderFindings(drift)}</TabsContent>
              <TabsContent value="stale" className="mt-0">{renderFindings(staleAndTerm)}</TabsContent>
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
