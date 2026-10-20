"""The local autoedit command-line interface."""
import argparse
from pathlib import Path
import sys

from .cache import write_json
from . import __version__
from .config import Config, load_config
from .probe import MediaError, require_tools


def build_parser():
    parser = argparse.ArgumentParser(prog="autoedit", description="Automatically edit source footage to music beats using visual and motion matching.")
    parser.add_argument("--version", action="version", version=f"autoedit {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "analyze"):
        sub = commands.add_parser(command)
        sub.add_argument("--audio", type=Path, required=True)
        sub.add_argument("--clips", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--config", type=Path)
        sub.add_argument("--duration", type=float)
        sub.add_argument("--audio-start", type=float)
        sub.add_argument("--pacing", choices=("slow", "medium", "fast"))
        sub.add_argument("--beat-sensitivity", type=float, help="0.5..3.0; higher values detect weaker onsets and favor more frequent beat cuts (default: 1)")
        sub.add_argument("--kick-focus", action="store_true", default=None, help="cut on kick-drum beats instead of every detected beat")
        sub.add_argument("--quiet-threshold", type=float, help="hold shots in quiet sections below this fraction of foreground RMS energy; 0 disables (default: 0.6)")
        sizing = sub.add_mutually_exclusive_group()
        sizing.add_argument("--resolution", help="WIDTHxHEIGHT; stretch each clip to exactly fill these output dimensions")
        sizing.add_argument("--size", help="legacy canvas WIDTHxHEIGHT or auto, honoring --fit (default: auto from input clips)")
        sub.add_argument("--fps", type=int)
        sub.add_argument("--fit", choices=("contain", "cover", "stretch"))
        sub.add_argument("--speed-min", type=float)
        sub.add_argument("--speed-max", type=float)
        sub.add_argument("--reuse", choices=("allow", "forbid"))
        sub.add_argument("--seed", type=int)
        sub.add_argument("--cache-dir")
        sub.add_argument("--overwrite", action="store_true")
    sub = commands.add_parser("render", help="replay a saved selection without analysis or search")
    sub.add_argument("--plan", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--overwrite", action="store_true")
    sub = commands.add_parser("validate")
    sub.add_argument("--plan", type=Path, required=True)
    sub.add_argument("--video", type=Path, required=True)
    sub = commands.add_parser("review", help="open the offline interactive beat editor for a saved plan")
    sub.add_argument("--plan", type=Path, required=True)
    sub.add_argument("--session", type=Path, help="saved timing draft (default: NAME.review.json next to the plan)")
    sub.add_argument("--video", type=Path, help="existing edit for the initial preview (default: NAME.mp4 next to the plan)")
    sub.add_argument("--port", type=int, default=8765, help="localhost port; 0 chooses a free port (default: 8765)")
    sub.add_argument("--no-browser", action="store_true", help="print the editor URL without opening a browser")
    return parser


def _guard_outputs(output, overwrite, sidecars=True):
    destinations = [output]
    if sidecars:
        destinations += [output.with_suffix(suffix) for suffix in (".plan.json", ".validation.json", ".report.html")]
    for path in destinations:
        if path.exists() and not overwrite:
            raise MediaError(f"destination already exists: {path}; use --overwrite")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        require_tools()
        if args.command == "review":
            if not 0 <= args.port <= 65535:
                raise ValueError("port must be between 0 and 65535")
            from .review import serve_review
            return serve_review(args.plan, session_path=args.session, video_path=args.video,
                                port=args.port, open_browser=not args.no_browser)
        if args.command in {"create", "analyze"}:
            from .pipeline import analyze_inputs, build_plan, versions
            from .video import serializable_analysis
            overrides = {k: getattr(args, k, None) for k in Config.__dataclass_fields__}
            if args.resolution is not None:
                if args.resolution.lower() == "auto":
                    raise ValueError("--resolution requires explicit WIDTHxHEIGHT dimensions")
                overrides.update(size=args.resolution, fit="stretch")
            cfg = load_config(args.config, overrides)
            output = args.output.expanduser().resolve()
            _guard_outputs(output, args.overwrite, sidecars=args.command == "create")
            if args.command == "create" and output.suffix.lower() != ".mp4":
                raise MediaError("output must have .mp4 extension")
            audio, clips, skipped = analyze_inputs(args.audio, args.clips, cfg, exclude=(output,))
            if args.command == "analyze":
                write_json(output, {"version": 1, "versions": versions(), "configuration": cfg.to_dict(),
                                    "audio": audio, "sources": [serializable_analysis(c) for c in clips], "skipped_sources": skipped})
                print(f"Analysis: {output}")
                return 0
            print("Planning beat schedule and visual sequence…", flush=True)
            plan = build_plan(audio, clips, cfg, skipped)
        elif args.command == "render":
            from .render import load_plan
            plan = load_plan(args.plan)
            output = args.output.expanduser().resolve()
            # Reading a plan next to a missing output is a normal replay workflow.
            if output.with_suffix(".plan.json") != args.plan.resolve():
                _guard_outputs(output, args.overwrite)
            elif output.exists() and not args.overwrite:
                raise MediaError(f"output already exists: {output}; use --overwrite")
        else:
            from .render import load_plan
            from .validate import validate_video
            plan = load_plan(args.plan)
            report = validate_video(plan, args.video)
            write_json(args.video.with_suffix(".validation.json"), report)
            print(f"Validation {'passed' if report['valid'] else 'FAILED'}: {report['decoded_frames']} decoded frames; "
                  f"beat error {report['max_beat_error_seconds']*1000:.2f} ms")
            for error in report["errors"]:
                print(error, file=sys.stderr)
            return 0 if report["valid"] else 1
        from .render import render_plan
        from .report import write_report
        print(f"Rendering {len(plan['segments'])} shots, {plan['output']['total_frames']} frames…", flush=True)
        validation = render_plan(plan, output, args.overwrite)
        write_json(output.with_suffix(".plan.json"), plan)
        write_report(plan, output, validation, output.with_suffix(".report.html"))
        print(f"Edit: {output}\nPlan: {output.with_suffix('.plan.json')}\nReport: {output.with_suffix('.report.html')}")
        return 0
    except (MediaError, ValueError, OSError) as exc:
        print(f"autoedit: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
