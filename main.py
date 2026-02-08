import argparse
from datetime import datetime
import logging
from pathlib import Path
import re
import shutil
import sys
import textwrap
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
    return max(80, min(width, 140))


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


def render_banner(style: CLIStyle):
    lines = [
        style.color("███╗   ███╗ █████╗ ██╗  ██╗███████╗", CLIStyle.BOLD, CLIStyle.TEAL),
        style.color("████╗ ████║██╔══██╗██║ ██╔╝██╔════╝", CLIStyle.BOLD, CLIStyle.TEAL),
        style.color("██╔████╔██║███████║█████╔╝ █████╗  ", CLIStyle.BOLD, CLIStyle.TEAL),
        style.color("██║╚██╔╝██║██╔══██║██╔═██╗ ██╔══╝  ", CLIStyle.BOLD, CLIStyle.TEAL),
        style.color("██║ ╚═╝ ██║██║  ██║██║  ██╗███████╗", CLIStyle.BOLD, CLIStyle.TEAL),
        style.color("╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝", CLIStyle.BOLD, CLIStyle.TEAL),
        style.color("IT HEAVY", CLIStyle.BOLD, CLIStyle.GOLD),
    ]
    render_panel(style, "Launch", lines)


def wrap_response_text(text: str, width: int):
    wrapped_lines = []
    for paragraph in (text or "").splitlines():
        raw = paragraph.rstrip()
        if not raw:
            wrapped_lines.append("")
            continue

        bullet_match = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$", raw)
        if bullet_match:
            prefix = bullet_match.group(1)
            body = bullet_match.group(2).strip()
            fill = textwrap.fill(
                body,
                width=max(40, width - len(prefix)),
                initial_indent=prefix,
                subsequent_indent=" " * len(prefix),
            )
            wrapped_lines.extend(fill.splitlines())
        else:
            fill = textwrap.fill(raw, width=width)
            wrapped_lines.extend(fill.splitlines())
    return wrapped_lines


def create_session_log(provider_info: dict) -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"session_{timestamp}.md"
    header = [
        "# Make It Heavy Session",
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
        "### Assistant",
        "",
        response.strip(),
        "",
        "---",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(section))


def show_provider_list(style: CLIStyle):
    title = style.color("Available Providers", CLIStyle.BOLD, CLIStyle.GOLD)
    lines = [title, ""]
    for provider_name in ProviderFactory.get_available_providers():
        info = ProviderFactory.get_provider_info(provider_name)
        label = f"{info['display_name']} ({provider_name})"
        lines.append(style.color(label, CLIStyle.BOLD, CLIStyle.CYAN))
        lines.append(f"  {info['description']}")
        lines.append(style.color(f"  Default model: {info['default_model']}", CLIStyle.DIM, CLIStyle.GRAY))
        lines.append("")
    render_panel(style, "Providers", lines[:-1] if lines and lines[-1] == "" else lines)


def main():
    """Main entry point for the AI agent with provider selection."""
    parser = argparse.ArgumentParser(description="AI Agent with multi-provider support")
    parser.add_argument(
        "--provider",
        choices=ProviderFactory.get_available_providers(),
        help="AI provider to use (overrides config.yaml)",
    )
    parser.add_argument("--list-providers", action="store_true", help="List available providers and exit")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show internal agent/tool debug logs",
    )
    args = parser.parse_args()

    style = CLIStyle(enabled=sys.stdout.isatty())

    if not args.verbose:
        logging.getLogger("providers.groq_provider").setLevel(logging.CRITICAL)

    if args.list_providers:
        show_provider_list(style)
        return

    try:
        from agent import AIAgent

        agent = AIAgent(provider_name=args.provider, silent=not args.verbose)
        provider_info = agent.provider_info
        session_log_path = create_session_log(provider_info)

        render_banner(style)
        intro_lines = [
            style.color("MAKE IT HEAVY", CLIStyle.BOLD, CLIStyle.GOLD),
            style.color("Multi-Provider Research Agent", CLIStyle.DIM, CLIStyle.GRAY),
            "",
            f"{style.color('Provider:', CLIStyle.BOLD, CLIStyle.CYAN)} {provider_info['display_name']}",
            f"{style.color('Model:', CLIStyle.BOLD, CLIStyle.CYAN)} {provider_info['model']}",
            f"{style.color('Mode:', CLIStyle.BOLD, CLIStyle.CYAN)} {'Verbose' if args.verbose else 'Professional'}",
            f"{style.color('Log:', CLIStyle.BOLD, CLIStyle.CYAN)} {session_log_path}",
            "",
            style.color("Type 'quit', 'exit', or 'bye' to exit.", CLIStyle.DIM, CLIStyle.GRAY),
        ]
        render_panel(style, "Session", intro_lines)

    except ModuleNotFoundError as e:
        print(style.color(f"Missing dependency: {e}", CLIStyle.BOLD, CLIStyle.RED))
        print("Install dependencies with: pip install -r requirements.txt")
        return
    except Exception as e:
        print(style.color(f"Error initializing agent: {e}", CLIStyle.BOLD, CLIStyle.RED))
        print()
        print("Make sure you have:")
        print("1. Set your API keys in config.yaml for the selected provider")
        print("2. Installed all dependencies with: pip install -r requirements.txt")
        print()
        print("To see available providers, run: python main.py --list-providers")
        return

    prompt_label = style.color("You ❯ ", CLIStyle.BOLD, CLIStyle.CYAN) if style.use_unicode else style.color("You > ", CLIStyle.BOLD, CLIStyle.CYAN)
    while True:
        try:
            user_input = input(f"\n{prompt_label}").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print(style.color("Session closed.", CLIStyle.BOLD, CLIStyle.GRAY))
                break

            if not user_input:
                print(style.color("Please enter a question or command.", CLIStyle.DIM, CLIStyle.GRAY))
                continue

            print(style.color("Assistant is analyzing...", CLIStyle.DIM, CLIStyle.BLUE))
            response = agent.run(user_input)
            append_exchange_to_log(session_log_path, user_input, response)

            response_lines = wrap_response_text(response, width=terminal_width() - 8)
            if not response_lines:
                response_lines = [style.color("No response generated.", CLIStyle.RED)]
            render_panel(style, "Assistant Response", response_lines)

        except KeyboardInterrupt:
            print("\n" + style.color("Session interrupted.", CLIStyle.DIM, CLIStyle.GRAY))
            break
        except Exception as e:
            print(style.color(f"Error: {e}", CLIStyle.BOLD, CLIStyle.RED))
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main()
