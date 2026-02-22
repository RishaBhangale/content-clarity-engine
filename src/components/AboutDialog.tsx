import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogClose,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface AboutDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const AboutDialog = ({ open, onOpenChange }: AboutDialogProps) => (
  <Dialog open={open} onOpenChange={onOpenChange}>
    <DialogContent className="sm:max-w-md">
      <DialogHeader>
        <DialogTitle className="text-xl">About C1</DialogTitle>
        <DialogDescription className="text-foreground-body leading-relaxed pt-2">
          C1 is an AI-powered Content Consistency &amp; Contradiction Detection engine for enterprise
          teams. It scans your document corpus to surface direct contradictions, semantic drift, stale
          references, and terminology inconsistencies — so you can trust your documentation. This is a
          demo showcasing the core workflow.
        </DialogDescription>
      </DialogHeader>
      <DialogClose asChild>
        <Button variant="outline" className="mt-2">Close</Button>
      </DialogClose>
    </DialogContent>
  </Dialog>
);

export default AboutDialog;
