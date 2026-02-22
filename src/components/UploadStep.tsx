import { useState, useCallback, useRef } from "react";
import { Upload, X, FileText, FileType, File } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { UploadedFile, SAMPLE_FILES, formatFileSize, getTotalSize } from "@/data/sampleData";
import { toast } from "sonner";

interface UploadStepProps {
  files: UploadedFile[];
  setFiles: (files: UploadedFile[]) => void;
  onContinue: () => void;
}

const FILE_ICONS: Record<string, React.ReactNode> = {
  PDF: <FileText className="w-5 h-5 text-destructive" />,
  DOCX: <FileType className="w-5 h-5 text-info" />,
  MD: <File className="w-5 h-5 text-foreground-muted" />,
  TXT: <File className="w-5 h-5 text-foreground-muted" />,
  HTML: <File className="w-5 h-5 text-warning" />,
};

const UploadStep = ({ files, setFiles, onContinue }: UploadStepProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => setIsDragging(false), []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const droppedFiles = Array.from(e.dataTransfer.files).map((f, i) => ({
        id: `upload-${Date.now()}-${i}`,
        name: f.name,
        size: f.size,
        type: f.name.split(".").pop()?.toUpperCase() || "FILE",
      }));
      setFiles([...files, ...droppedFiles]);
    },
    [files, setFiles]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files) return;
      const selected = Array.from(e.target.files).map((f, i) => ({
        id: `upload-${Date.now()}-${i}`,
        name: f.name,
        size: f.size,
        type: f.name.split(".").pop()?.toUpperCase() || "FILE",
      }));
      setFiles([...files, ...selected]);
      e.target.value = "";
    },
    [files, setFiles]
  );

  const removeFile = useCallback(
    (id: string) => setFiles(files.filter((f) => f.id !== id)),
    [files, setFiles]
  );

  const loadSamples = useCallback(() => {
    setFiles(SAMPLE_FILES);
    toast.success("Sample documents loaded");
  }, [setFiles]);

  return (
    <div className="max-w-3xl mx-auto px-4">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">Upload Your Documents</h1>
        <p className="text-foreground-muted max-w-xl mx-auto">
          Drop your document files below to begin scanning for contradictions, semantic drift, stale
          references, and terminology inconsistencies.
        </p>
      </div>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative min-h-[280px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors duration-200 ${
          isDragging
            ? "border-primary bg-primary-light"
            : "border-border-dashed bg-background-subtle hover:border-primary/40"
        }`}
      >
        <Upload className={`w-12 h-12 ${isDragging ? "text-primary" : "text-foreground-muted"}`} />
        <p className="text-lg font-medium">{isDragging ? "Drop files here" : "Drag & drop files here"}</p>
        <p className="text-sm text-foreground-muted">or click to browse</p>
        <div className="flex gap-2 mt-2">
          {["PDF", "DOCX", "HTML", "MD", "TXT"].map((fmt) => (
            <Badge key={fmt} variant="secondary" className="text-xs">
              {fmt}
            </Badge>
          ))}
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.docx,.html,.md,.txt"
          onChange={handleFileSelect}
          className="hidden"
        />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <div className="mt-6 border border-border rounded-lg overflow-hidden">
          {files.map((file) => (
            <div
              key={file.id}
              className="flex items-center justify-between px-4 py-3 border-b border-border last:border-b-0 hover:bg-background-subtle transition-colors"
            >
              <div className="flex items-center gap-3">
                {FILE_ICONS[file.type] || <File className="w-5 h-5 text-foreground-muted" />}
                <span className="text-sm font-medium">{file.name}</span>
                <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                  {file.type}
                </Badge>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-xs text-foreground-muted">{formatFileSize(file.size)}</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeFile(file.id);
                  }}
                  className="text-destructive hover:text-destructive/80 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
          <div className="px-4 py-2.5 bg-background-subtle text-xs text-foreground-muted">
            {files.length} file{files.length !== 1 ? "s" : ""} selected · Total size: {getTotalSize(files)}
          </div>
        </div>
      )}

      {/* Sample docs */}
      <div className="mt-6">
        <Separator className="mb-4" />
        <div className="text-center">
          <button
            onClick={loadSamples}
            className="text-sm text-primary font-medium hover:underline"
          >
            Or try with sample documents →
          </button>
        </div>
      </div>

      {/* Continue */}
      <div className="flex justify-center mt-8">
        <Button
          size="lg"
          disabled={files.length === 0}
          onClick={onContinue}
          className="w-full max-w-md"
        >
          Continue to Configuration →
        </Button>
      </div>
    </div>
  );
};

export default UploadStep;
