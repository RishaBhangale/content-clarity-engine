import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { ChevronDown, ChevronUp, Lightbulb } from "lucide-react";
import { Finding } from "@/data/sampleData";

interface FindingCardProps {
  finding: Finding;
}

const SEVERITY_STYLES = {
  critical: "bg-destructive-light text-destructive border-destructive/20",
  warning: "bg-warning-light text-warning-foreground border-warning/20",
  info: "bg-info-light text-info border-info/20",
};

const SEVERITY_LABELS = {
  critical: "Critical",
  warning: "Warning",
  info: "Info",
};

const TYPE_STYLES = {
  Contradiction: "border-destructive/30 text-destructive",
  "Semantic Drift": "border-warning/30 text-warning-foreground",
  "Stale Reference": "border-info/30 text-info",
  Terminology: "border-primary/30 text-primary",
};

const FindingCard = ({ finding }: FindingCardProps) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="border border-border rounded-lg overflow-hidden bg-card hover:shadow-sm transition-shadow cursor-pointer"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start gap-3 p-4">
        <Badge className={`${SEVERITY_STYLES[finding.severity]} text-xs shrink-0 mt-0.5`}>
          {SEVERITY_LABELS[finding.severity]}
        </Badge>
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-semibold">{finding.title}</h4>
          <p className="text-xs text-foreground-muted mt-1">
            {finding.sourceA}
            {finding.sourceB ? ` ↔ ${finding.sourceB}` : ""}
          </p>
        </div>
        <Badge variant="outline" className={`${TYPE_STYLES[finding.type]} text-[10px] shrink-0`}>
          {finding.type}
        </Badge>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-foreground-muted shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-foreground-muted shrink-0" />
        )}
      </div>

      {expanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-border pt-3">
          <div className="bg-background-subtle rounded-md p-3 border-l-2 border-destructive/40">
            <p className="text-[10px] font-medium text-foreground-muted uppercase tracking-wider mb-1">
              {finding.sourceA.split("§")[0]?.trim()}
            </p>
            <p className="text-sm text-foreground-body italic">{finding.excerptA}</p>
          </div>

          {finding.excerptB && (
            <div className="bg-background-subtle rounded-md p-3 border-l-2 border-warning/40">
              <p className="text-[10px] font-medium text-foreground-muted uppercase tracking-wider mb-1">
                {finding.sourceB?.split("§")[0]?.trim() || "Source B"}
              </p>
              <p className="text-sm text-foreground-body italic">{finding.excerptB}</p>
            </div>
          )}

          <div className="bg-secondary-light rounded-md p-3 border border-secondary/20 flex items-start gap-2">
            <Lightbulb className="w-4 h-4 text-secondary shrink-0 mt-0.5" />
            <div>
              <p className="text-[10px] font-medium text-secondary uppercase tracking-wider mb-1">Suggested Resolution</p>
              <p className="text-sm text-foreground-body">{finding.suggestion}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-foreground-muted">Confidence</span>
            <Progress value={finding.confidence} className="h-1.5 flex-1 max-w-[120px]" />
            <span className="text-xs font-medium">{finding.confidence}%</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default FindingCard;
