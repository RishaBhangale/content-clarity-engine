import { Button } from "@/components/ui/button";
import AboutDialog from "./AboutDialog";
import { useState } from "react";
import { RotateCcw } from "lucide-react";

interface NavbarProps {
  onReset: () => void;
}

const Navbar = ({ onReset }: NavbarProps) => {
  const [aboutOpen, setAboutOpen] = useState(false);

  return (
    <>
      <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="max-w-7xl mx-auto flex h-14 items-center justify-between px-4 sm:px-6">
          <div className="flex items-center gap-2">
            <span className="text-xl font-extrabold text-primary tracking-tight">
              C1
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-secondary ml-0.5 mb-2" />
            </span>
            <span className="text-sm text-foreground-muted hidden sm:inline">
              Content Consistency Engine
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={() => setAboutOpen(true)}>
              About
            </Button>
            <Button variant="ghost" size="sm" onClick={onReset} className="gap-1.5">
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Demo
            </Button>
          </div>
        </div>
      </header>
      <AboutDialog open={aboutOpen} onOpenChange={setAboutOpen} />
    </>
  );
};

export default Navbar;
