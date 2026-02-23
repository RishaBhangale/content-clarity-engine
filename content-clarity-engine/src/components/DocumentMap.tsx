import { UploadedFile, Finding } from "@/data/sampleData";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface DocumentMapProps {
  files: UploadedFile[];
  findings: Finding[];
}

function getDocFindings(fileName: string, findings: Finding[]): Finding[] {
  return findings.filter(
    (f) =>
      f.sourceA.includes(fileName.replace(/\.[^.]+$/, "").replace(/-/g, "-")) ||
      (f.sourceB && f.sourceB.includes(fileName.replace(/\.[^.]+$/, "").replace(/-/g, "-")))
  );
}

function getShortName(name: string) {
  return name.replace(/\.[^.]+$/, "").replace(/-/g, " ").replace(/v\d+$/, "").trim();
}

const SEVERITY_COLORS = {
  critical: "bg-destructive",
  warning: "bg-warning",
  info: "bg-info",
};

const DocumentMap = ({ files, findings }: DocumentMapProps) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Document Map</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {files.map((file) => {
            const docFindings = getDocFindings(file.name, findings);
            const maxSeverity = docFindings.some((f) => f.severity === "critical")
              ? "critical"
              : docFindings.some((f) => f.severity === "warning")
              ? "warning"
              : "info";

            return (
              <div
                key={file.id}
                className={`border rounded-lg p-3 text-center bg-background-subtle hover:shadow-sm transition-shadow ${
                  docFindings.length > 0 ? `border-l-2` : ""
                }`}
                style={
                  docFindings.length > 0
                    ? {
                        borderLeftColor:
                          maxSeverity === "critical"
                            ? "hsl(var(--destructive))"
                            : maxSeverity === "warning"
                            ? "hsl(var(--warning))"
                            : "hsl(var(--info))",
                      }
                    : undefined
                }
              >
                <p className="text-xs font-medium truncate">{getShortName(file.name)}</p>
                <p className="text-[10px] text-foreground-muted mt-0.5">{file.type}</p>
                {docFindings.length > 0 && (
                  <Badge variant="outline" className="mt-2 text-[10px]">
                    {docFindings.length} finding{docFindings.length !== 1 ? "s" : ""}
                  </Badge>
                )}
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
};

export default DocumentMap;
