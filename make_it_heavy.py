import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import textwrap
import threading
import time
from providers import ProviderFactory


class CLIStyle:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[38;5;45m"
    BLUE = "\033[38;5;39m"
    GREEN = "\033[38;5;82m"
    GOLD = "\033[38;5;220m"
    RED = "\033[38;5;196m"
    GRAY = "\033[38;5;246m"
    TEAL = "\033[38;5;44m"
    ORANGE = "\033[38;5;208m"

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.use_unicode = "utf" in (sys.stdout.encoding or "").lower()

    def color(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        return f"{''.join(styles)}{text}{self.RESET}"

    def box_chars(self):
        if self.use_unicode:
            return {
                "h": "═",
                "v": "║",
                "tl": "╔",
                "tr": "╗",
                "bl": "╚",
                "br": "╝",
            }
        return {
            "h": "-",
            "v": "|",
            "tl": "+",
            "tr": "+",
            "bl": "+",
            "br": "+",
        }


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def terminal_width(default: int = 100) -> int:
    width = shutil.get_terminal_size((default, 24)).columns
    return max(80, min(width, 160))


def render_panel(style: CLIStyle, title: str, lines):
    chars = style.box_chars()
    width = terminal_width()
    inner_width = width - 4

    title_text = f" {title} "
    top_core = title_text + chars["h"] * max(0, inner_width - len(strip_ansi(title_text)))
    print(style.color(chars["tl"] + top_core + chars["tr"], CLIStyle.BLUE))

    for line in lines:
        visible = len(strip_ansi(line))
        padded = line + (" " * max(0, inner_width - visible))
        print(style.color(chars["v"], CLIStyle.BLUE) + " " + padded + " " + style.color(chars["v"], CLIStyle.BLUE))

    print(style.color(chars["bl"] + (chars["h"] * inner_width) + chars["br"], CLIStyle.BLUE))


def wrap_response_text(text: str, width: int):
    wrapped_lines = []
    for paragraph in (text or "").splitlines():
        raw = paragraph.rstrip()
        if not raw:
            wrapped_lines.append("")
            continue
        fill = textwrap.fill(raw, width=max(40, width))
        wrapped_lines.extend(fill.splitlines())
    return wrapped_lines


def create_session_log(provider_info: dict) -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"heavy_session_{timestamp}.md"
    header = [
        "# Make It Heavy Orchestrator Session",
        "",
        f"- Started: {datetime.now().isoformat(timespec='seconds')}",
        f"- Provider: {provider_info.get('display_name', 'unknown')}",
        f"- Model: {provider_info.get('model', 'unknown')}",
        "",
        "---",
        "",
    ]
    path.write_text("\n".join(header), encoding="utf-8")
    return path


def append_exchange_to_log(log_path: Path, user_input: str, response: str):
    ts = datetime.now().isoformat(timespec="seconds")
    section = [
        f"## Exchange {ts}",
        "",
        "### User",
        "",
        user_input.strip(),
        "",
        "### Final Response",
        "",
        response.strip(),
        "",
        "---",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(section))


def should_log_successful_result(result: str) -> bool:
    if not result or not result.strip():
        return False
    normalized = result.strip().lower()
    failure_markers = [
        "all agents failed to provide meaningful results",
        "error:",
        "timed out",
        "task failed",
    ]
    return not any(marker in normalized for marker in failure_markers)


def show_provider_list(style: CLIStyle):
    lines = [style.color("Available Providers", CLIStyle.BOLD, CLIStyle.GOLD), ""]
    for provider_name in ProviderFactory.get_available_providers():
        info = ProviderFactory.get_provider_info(provider_name)
        lines.append(style.color(f"{info['display_name']} ({provider_name})", CLIStyle.BOLD, CLIStyle.CYAN))
        lines.append(f"  {info['description']}")
        lines.append(style.color(f"  Default model: {info['default_model']}", CLIStyle.DIM, CLIStyle.GRAY))
        lines.append("")
    render_panel(style, "Providers", lines[:-1] if lines and lines[-1] == "" else lines)


class OrchestratorCLI:
    def __init__(self, provider_name=None, verbose=False):
        from orchestrator import TaskOrchestrator

        self.verbose = verbose
        self.style = CLIStyle(enabled=sys.stdout.isatty())

        if not verbose:
            logging.getLogger("providers.groq_provider").setLevel(logging.CRITICAL)

        self.orchestrator = TaskOrchestrator(provider_name=provider_name, silent=not verbose)
        self.start_time = None
        self.running = False

        self.provider_info = self.orchestrator.provider.get_provider_info()
        self.session_log_path = create_session_log(self.provider_info)

    def clear_screen(self):
        os.system("cls" if os.name == "nt" else "clear")

    def format_time(self, seconds):
        if seconds < 60:
            return f"{int(seconds)}s"
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

    def create_progress_bar(self, status: str) -> str:
        width = max(18, min(56, terminal_width() - 52))
        status_upper = (status or "").upper()
        if status_upper == "COMPLETED":
            fill = width
            color = CLIStyle.GREEN
            marker = "■"
        elif "FAILED" in status_upper or "ERROR" in status_upper:
            fill = width
            color = CLIStyle.RED
            marker = "■"
        elif "RETRY" in status_upper:
            fill = max(3, width // 3)
            color = CLIStyle.ORANGE
            marker = "▣"
        elif status_upper == "PROCESSING...":
            fill = max(4, width // 2)
            color = CLIStyle.CYAN
            marker = "▣"
        elif status_upper == "TIMEOUT":
            fill = width
            color = CLIStyle.RED
            marker = "■"
        else:
            fill = 0
            color = CLIStyle.GRAY
            marker = "·"

        if self.style.use_unicode:
            bar = marker * fill + "·" * max(0, width - fill)
        else:
            bar = "#" * fill + "." * max(0, width - fill)
        return self.style.color(bar, color)

    def update_display(self, force=False):
        if not self.running and not force:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0
        progress = self.orchestrator.get_progress_status()

        self.clear_screen()
        header_lines = [
            self.style.color("MAKE IT HEAVY ORCHESTRATOR", CLIStyle.BOLD, CLIStyle.GOLD),
            self.style.color("Parallel Multi-Agent Research", CLIStyle.DIM, CLIStyle.GRAY),
            "",
            f"{self.style.color('Provider:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.provider_info['display_name']}",
            f"{self.style.color('Model:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.provider_info['model']}",
            f"{self.style.color('Agents:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.orchestrator.num_agents} "
            f"(max concurrency {self.orchestrator.max_concurrency})",
            f"{self.style.color('Elapsed:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.format_time(elapsed)}",
            f"{self.style.color('State:', CLIStyle.BOLD, CLIStyle.CYAN)} "
            f"{'Running' if self.running else 'Completed'}",
        ]
        render_panel(self.style, "Heavy Session", header_lines)

        agent_lines = []
        for i in range(self.orchestrator.num_agents):
            status = progress.get(i, "QUEUED")
            bar = self.create_progress_bar(status)
            agent_lines.append(
                f"{self.style.color(f'Agent {i + 1:02d}', CLIStyle.BOLD, CLIStyle.CYAN)} "
                f"{bar} "
                f"{self.style.color(status, CLIStyle.DIM, CLIStyle.GRAY)}"
            )
        render_panel(self.style, "Agent Progress", agent_lines)
        sys.stdout.flush()

    def progress_monitor(self):
        while self.running:
            self.update_display()
            time.sleep(0.4)

    def run_task(self, user_input: str):
        self.start_time = time.time()
        self.running = True
        progress_thread = threading.Thread(target=self.progress_monitor, daemon=True)
        progress_thread.start()

        try:
            result = self.orchestrator.orchestrate(user_input)
            self.running = False
            self.update_display(force=True)

            if should_log_successful_result(result or ""):
                append_exchange_to_log(self.session_log_path, user_input, result)

            response_lines = wrap_response_text(result or "", width=terminal_width() - 10)
            if not response_lines:
                response_lines = [self.style.color("No response generated.", CLIStyle.RED)]
            render_panel(self.style, "Final Result", response_lines)
            return result
        except Exception as e:
            self.running = False
            self.update_display(force=True)
            err_lines = wrap_response_text(str(e), width=terminal_width() - 10)
            render_panel(
                self.style,
                "Error",
                [self.style.color("Orchestration failed.", CLIStyle.BOLD, CLIStyle.RED), ""] + err_lines
            )
            return None

    def interactive_mode(self):
        intro_lines = [
            self.style.color("MAKE IT HEAVY", CLIStyle.BOLD, CLIStyle.GOLD),
            self.style.color("Multi-Agent Orchestrator", CLIStyle.DIM, CLIStyle.GRAY),
            "",
            f"{self.style.color('Provider:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.provider_info['display_name']}",
            f"{self.style.color('Model:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.provider_info['model']}",
            f"{self.style.color('Parallel agents:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.orchestrator.num_agents}",
            f"{self.style.color('Max concurrency:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.orchestrator.max_concurrency}",
            f"{self.style.color('Log:', CLIStyle.BOLD, CLIStyle.CYAN)} {self.session_log_path}",
            "",
            self.style.color("Type 'quit', 'exit', or 'bye' to exit.", CLIStyle.DIM, CLIStyle.GRAY),
        ]
        render_panel(self.style, "Ready", intro_lines)

        prompt_label = (
            self.style.color("You ❯ ", CLIStyle.BOLD, CLIStyle.CYAN)
            if self.style.use_unicode
            else self.style.color("You > ", CLIStyle.BOLD, CLIStyle.CYAN)
        )

        while True:
            try:
                user_input = input(f"\n{prompt_label}").strip()
                if user_input.lower() in ["quit", "exit", "bye"]:
                    print(self.style.color("Session closed.", CLIStyle.BOLD, CLIStyle.GRAY))
                    break
                if not user_input:
                    print(self.style.color("Please enter a question or command.", CLIStyle.DIM, CLIStyle.GRAY))
                    continue

                print(self.style.color("Heavy orchestration in progress...", CLIStyle.DIM, CLIStyle.BLUE))
                result = self.run_task(user_input)
                if result is None:
                    print(self.style.color("Task failed. Please try again.", CLIStyle.RED))

            except KeyboardInterrupt:
                print("\n" + self.style.color("Session interrupted.", CLIStyle.DIM, CLIStyle.GRAY))
                break
            except Exception as e:
                print(self.style.color(f"Error: {e}", CLIStyle.BOLD, CLIStyle.RED))
                print("Please try again or type 'quit' to exit.")


def main():
    parser = argparse.ArgumentParser(description="Multi-agent orchestrator with provider selection")
    parser.add_argument(
        "--provider",
        choices=ProviderFactory.get_available_providers(),
        help="AI provider to use (overrides config.yaml)",
    )
    parser.add_argument("--list-providers", action="store_true", help="List available providers and exit")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show internal agent/provider debug logs",
    )
    args = parser.parse_args()

    style = CLIStyle(enabled=sys.stdout.isatty())

    if args.list_providers:
        show_provider_list(style)
        return

    try:
        cli = OrchestratorCLI(provider_name=args.provider, verbose=args.verbose)
        cli.interactive_mode()
    except ModuleNotFoundError as e:
        print(style.color(f"Missing dependency: {e}", CLIStyle.BOLD, CLIStyle.RED))
        print("Install dependencies with: pip install -r requirements.txt")
    except Exception as e:
        print(style.color(f"Error: {e}", CLIStyle.BOLD, CLIStyle.RED))
        print()
        print("Make sure you have:")
        print("1. Set your API keys in config.yaml for the selected provider")
        print("2. Installed all dependencies with: pip install -r requirements.txt")
        print()
        print("To see available providers, run: python make_it_heavy.py --list-providers")


if __name__ == "__main__":
    main()
