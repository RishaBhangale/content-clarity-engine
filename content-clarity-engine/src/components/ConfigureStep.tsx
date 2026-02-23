import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Separator } from "@/components/ui/separator";
import { ScanConfig, UploadedFile, formatFileSize, getTotalSize } from "@/data/sampleData";

interface ConfigureStepProps {
  files: UploadedFile[];
  config: ScanConfig;
  setConfig: (config: ScanConfig) => void;
  onBack: () => void;
  onStart: () => void;
}

const DETECTION_TYPES = [
  { key: "contradiction" as const, label: "Direct Contradiction Detection", desc: "Find statements that directly conflict across documents" },
  { key: "semanticDrift" as const, label: "Semantic Drift Analysis", desc: "Identify processes described differently across document versions" },
  { key: "staleReference" as const, label: "Stale Reference Detection", desc: "Flag references to deprecated systems or outdated versions" },
  { key: "terminology" as const, label: "Terminology Inconsistency", desc: "Detect different terms used for the same concept across teams" },
];

const SENSITIVITY_OPTIONS = [
  { value: "high" as const, label: "High", desc: "Flag all potential issues including low-confidence matches" },
  { value: "medium" as const, label: "Medium", desc: "Balanced detection with reasonable confidence threshold" },
  { value: "low" as const, label: "Low", desc: "Only flag high-confidence contradictions" },
];

const SCOPE_OPTIONS = [
  { value: "cross" as const, label: "Cross-document", desc: "Compare every document against every other document" },
  { value: "within" as const, label: "Within-document", desc: "Check each document internally only" },
  { value: "both" as const, label: "Both", desc: "Run both cross-document and within-document analysis" },
];

const ConfigureStep = ({ files, config, setConfig, onBack, onStart }: ConfigureStepProps) => {
  const toggleDetection = (key: keyof ScanConfig["detectionTypes"]) => {
    setConfig({
      ...config,
      detectionTypes: { ...config.detectionTypes, [key]: !config.detectionTypes[key] },
    });
  };

  const enabledTypes = Object.entries(config.detectionTypes)
    .filter(([, v]) => v)
    .map(([k]) => k);

  const estimatedTime = files.length * 5;

  return (
    <div className="max-w-5xl mx-auto px-4">
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Left: Settings */}
        <div className="lg:col-span-3">
          <Card>
            <CardHeader>
              <CardTitle>Scan Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Detection Types */}
              <div>
                <h3 className="text-sm font-semibold mb-3">Detection Types</h3>
                <div className="space-y-3">
                  {DETECTION_TYPES.map((dt) => (
                    <div key={dt.key} className="flex items-start gap-3">
                      <Checkbox
                        id={dt.key}
                        checked={config.detectionTypes[dt.key]}
                        onCheckedChange={() => toggleDetection(dt.key)}
                        className="mt-0.5"
                      />
                      <Label htmlFor={dt.key} className="cursor-pointer">
                        <span className="text-sm font-medium">{dt.label}</span>
                        <p className="text-xs text-foreground-muted mt-0.5">{dt.desc}</p>
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              {/* Sensitivity */}
              <div>
                <h3 className="text-sm font-semibold mb-3">Sensitivity Level</h3>
                <RadioGroup
                  value={config.sensitivity}
                  onValueChange={(v) => setConfig({ ...config, sensitivity: v as ScanConfig["sensitivity"] })}
                  className="space-y-2"
                >
                  {SENSITIVITY_OPTIONS.map((opt) => (
                    <div key={opt.value} className="flex items-start gap-3">
                      <RadioGroupItem value={opt.value} id={`sens-${opt.value}`} className="mt-0.5" />
                      <Label htmlFor={`sens-${opt.value}`} className="cursor-pointer">
                        <span className="text-sm font-medium">{opt.label}</span>
                        <p className="text-xs text-foreground-muted mt-0.5">{opt.desc}</p>
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>

              <Separator />

              {/* Scope */}
              <div>
                <h3 className="text-sm font-semibold mb-3">Scope</h3>
                <RadioGroup
                  value={config.scope}
                  onValueChange={(v) => setConfig({ ...config, scope: v as ScanConfig["scope"] })}
                  className="space-y-2"
                >
                  {SCOPE_OPTIONS.map((opt) => (
                    <div key={opt.value} className="flex items-start gap-3">
                      <RadioGroupItem value={opt.value} id={`scope-${opt.value}`} className="mt-0.5" />
                      <Label htmlFor={`scope-${opt.value}`} className="cursor-pointer">
                        <span className="text-sm font-medium">{opt.label}</span>
                        <p className="text-xs text-foreground-muted mt-0.5">{opt.desc}</p>
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right: Summary */}
        <div className="lg:col-span-2">
          <Card className="sticky top-20">
            <CardHeader>
              <CardTitle className="text-base">Scan Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div>
                <p className="text-foreground-muted text-xs mb-1">Documents</p>
                <p className="font-medium">{files.length} files ({getTotalSize(files)})</p>
                <div className="mt-2 space-y-1">
                  {files.map((f) => (
                    <div key={f.id} className="flex items-center gap-2 text-xs">
                      <Badge variant="outline" className="text-[10px] px-1.5 py-0">{f.type}</Badge>
                      <span className="truncate">{f.name}</span>
                    </div>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <p className="text-foreground-muted text-xs mb-1.5">Detection Types</p>
                <div className="flex flex-wrap gap-1.5">
                  {enabledTypes.map((t) => (
                    <Badge key={t} className="bg-secondary/10 text-secondary border-secondary/20 text-xs">
                      {t === "contradiction" ? "Contradictions" : t === "semanticDrift" ? "Semantic Drift" : t === "staleReference" ? "Stale Refs" : "Terminology"}
                    </Badge>
                  ))}
                </div>
              </div>

              <Separator />

              <div className="flex justify-between">
                <span className="text-foreground-muted text-xs">Sensitivity</span>
                <span className="text-xs font-medium capitalize">{config.sensitivity}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-foreground-muted text-xs">Est. scan time</span>
                <span className="text-xs font-medium">~{estimatedTime} seconds</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Actions */}
      <div className="flex justify-between mt-8 max-w-3xl mx-auto">
        <Button variant="outline" onClick={onBack}>← Back</Button>
        <Button onClick={onStart}>Start Scan →</Button>
      </div>
    </div>
  );
};

export default ConfigureStep;
