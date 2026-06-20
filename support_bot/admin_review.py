"""
Admin review interface for extracted_history and rules.md.
Posts regular messages in the support channel (not ephemeral) so edits
are reliable — message.edit() works instantly without Discord token quirks.
"""

import logging

import discord
from discord import ui

from langchain_core.messages import HumanMessage

from shared.config import ADMIN_USER_IDS, SUPPORT_CHANNEL_ID
from shared.llm import get_llm
from support_bot.rules import (
    add_section,
    db_delete_entry,
    db_fetch_all,
    db_increment_view,
    db_update_entry,
    delete_section,
    load_rules,
    parse_sections,
    replace_section,
    replace_subsection,
)

log = logging.getLogger("admin_review")


# ── Admin guard ───────────────────────────────────────────────────────────────

async def _check_admin(interaction: discord.Interaction) -> bool:
    if interaction.user.id in ADMIN_USER_IDS:
        return True
    await interaction.response.send_message(
        "You don't have permission to use this.", ephemeral=True
    )
    return False


# ── LLM cleanup ───────────────────────────────────────────────────────────────

CLEAN_PROMPT = """You are editing a knowledge base entry for a solar sales support system.
The admin provided a quick edit below. Clean it up:
- Fix grammar and spelling
- Keep all facts exactly as written — do not add, remove, or interpret
- Use plain prose, no bullet lists unless the original had them
- Keep it concise

Return ONLY the cleaned text, no explanation.

Admin input:
INPUT_PLACEHOLDER"""


async def llm_clean(text: str) -> str:
    llm = get_llm(model_override="deepseek/deepseek-v4-flash", temperature=0.1)
    prompt = CLEAN_PROMPT.replace("INPUT_PLACEHOLDER", text)
    try:
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        return resp.content.strip()
    except Exception as e:
        log.warning("[LLM] clean failed: %s — saving raw", e)
        return text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(text: str, limit: int = 1000) -> str:
    if not text:
        return "(empty)"
    return text if len(text) <= limit else text[:limit] + "…"


def _view_emoji(view_count: int) -> str:
    """Color-code by view count using emoji that render reliably in Discord."""
    if view_count == 0:
        return "🔘"   # never seen
    if view_count <= 3:
        return "🔶"   # seen a few times
    return "✅"        # well reviewed


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTED HISTORY REVIEW
# ══════════════════════════════════════════════════════════════════════════════

PAGE_SIZE = 24  # leave 1 slot for "Show more…" when needed


def _build_list_embed(entries: list[dict], page: int, total_pages: int) -> discord.Embed:
    embed = discord.Embed(
        title="📋 Knowledge Review",
        description=(
            "Select a topic from the dropdown to view details.\n"
            "🔘 Never viewed  🔶 Seen 1–3×  ✅ Seen 4+×"
        ),
        color=discord.Color.blurple(),
    )
    embed.set_footer(text=f"Page {page + 1} of {total_pages} | {len(entries)} entries shown")
    return embed


def _build_detail_embed(entry: dict) -> discord.Embed:
    vc = entry["view_count"]
    emoji = _view_emoji(vc)
    embed = discord.Embed(
        title=f"{emoji} {entry['topic'] or '(no topic)'}",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="Summary", value=_truncate(entry["summary"], 900), inline=False)
    embed.add_field(name="Conversation (read-only)", value=_truncate(entry["conversation"], 900), inline=False)
    embed.set_footer(text=f"ID: {entry['id']} | Date: {entry['source_date']} | Viewed {vc}×")
    return embed


# ── Modals ────────────────────────────────────────────────────────────────────

class EditTopicModal(ui.Modal, title="Edit Topic"):
    def __init__(self, entry: dict, view: "HistoryDetailView"):
        super().__init__()
        self._entry = entry
        self._view = view
        self.topic_input = ui.TextInput(
            label="Topic",
            default=entry["topic"] or "",
            max_length=200,
            style=discord.TextStyle.short,
        )
        self.add_item(self.topic_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        cleaned = await llm_clean(self.topic_input.value)
        db_update_entry(self._entry["id"], topic=cleaned)
        self._entry["topic"] = cleaned
        embed = _build_detail_embed(self._entry)
        try:
            await self._view.message.edit(embed=embed, view=self._view)
            log.info("[review] topic updated for entry %s -> %r", self._entry["id"], cleaned)
        except Exception as e:
            log.error("[review] EditTopicModal message.edit failed: %s", e)


class EditSummaryModal(ui.Modal, title="Edit Summary"):
    def __init__(self, entry: dict, view: "HistoryDetailView"):
        super().__init__()
        self._entry = entry
        self._view = view
        self.summary_input = ui.TextInput(
            label="Summary",
            default=_truncate(entry["summary"] or "", 3900),
            max_length=4000,
            style=discord.TextStyle.paragraph,
        )
        self.add_item(self.summary_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        cleaned = await llm_clean(self.summary_input.value)
        db_update_entry(self._entry["id"], summary=cleaned)
        self._entry["summary"] = cleaned
        embed = _build_detail_embed(self._entry)
        try:
            await self._view.message.edit(embed=embed, view=self._view)
            log.info("[review] summary updated for entry %s", self._entry["id"])
        except Exception as e:
            log.error("[review] EditSummaryModal message.edit failed: %s", e)


# ── Topic dropdown ────────────────────────────────────────────────────────────

class TopicSelect(ui.Select):
    """
    Shows up to PAGE_SIZE topics + optional "Show more..." option.
    Ordered by view_count DESC so most-reviewed float to top.
    """
    def __init__(self, all_entries: list[dict], page: int, parent_view: "HistoryListView"):
        self._all_entries = all_entries
        self._page = page
        self._parent_view = parent_view

        start = page * PAGE_SIZE
        slice_ = all_entries[start: start + PAGE_SIZE]
        has_more = len(all_entries) > start + PAGE_SIZE

        options = []
        for e in slice_:
            emoji = _view_emoji(e["view_count"])
            label = f"{emoji} {e['topic'] or '(no topic)'}"[:100]
            options.append(discord.SelectOption(label=label, value=str(e["id"])))

        if has_more:
            options.append(discord.SelectOption(
                label="Show more...",
                value="__more__",
                description=f"Showing {start + 1}-{start + PAGE_SIZE} of {len(all_entries)}",
            ))

        super().__init__(placeholder="Select a topic to view...", options=options)

    async def callback(self, interaction: discord.Interaction):
        if not await _check_admin(interaction):
            return

        chosen = self.values[0]

        if chosen == "__more__":
            await interaction.response.defer()
            next_page = self._page + 1
            new_view = HistoryListView(self._all_entries, page=next_page)
            new_view.message = self._parent_view.message
            total_pages = -(-len(self._all_entries) // PAGE_SIZE)
            embed = _build_list_embed(self._all_entries, next_page, total_pages)
            try:
                await self._parent_view.message.edit(embed=embed, view=new_view)
            except Exception as e:
                log.error("[review] TopicSelect show-more edit failed: %s", e)
            return

        entry_id = int(chosen)
        entry = next((e for e in self._all_entries if e["id"] == entry_id), None)
        if not entry:
            await interaction.response.defer()
            return

        await interaction.response.defer()
        db_increment_view(entry_id)
        entry["view_count"] += 1

        detail_view = HistoryDetailView(entry, self._all_entries, self._page)
        detail_view.message = self._parent_view.message
        embed = _build_detail_embed(entry)
        try:
            await self._parent_view.message.edit(embed=embed, view=detail_view)
        except Exception as e:
            log.error("[review] TopicSelect detail edit failed: %s", e)


# ── List view ─────────────────────────────────────────────────────────────────

class HistoryListView(ui.View):
    def __init__(self, all_entries: list[dict], page: int = 0):
        super().__init__(timeout=None)
        self.all_entries = all_entries
        self.page = page
        self.message: discord.Message | None = None
        self.add_item(TopicSelect(all_entries, page, self))

    @ui.button(label="Back", style=discord.ButtonStyle.secondary, row=1)
    async def prev_page(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        if self.page == 0:
            await interaction.response.defer()
            return
        await interaction.response.defer()
        new_page = self.page - 1
        new_view = HistoryListView(self.all_entries, page=new_page)
        new_view.message = self.message
        total_pages = -(-len(self.all_entries) // PAGE_SIZE)
        embed = _build_list_embed(self.all_entries, new_page, total_pages)
        try:
            await self.message.edit(embed=embed, view=new_view)
        except Exception as e:
            log.error("[review] prev_page edit failed: %s", e)

    @ui.button(label="✖ Close", style=discord.ButtonStyle.danger, row=1)
    async def close(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        await interaction.response.defer()
        try:
            await self.message.delete()
        except Exception as e:
            log.error("[review] HistoryListView close failed: %s", e)


# ── Detail view ───────────────────────────────────────────────────────────────

class HistoryDetailView(ui.View):
    def __init__(self, entry: dict, all_entries: list[dict], list_page: int):
        super().__init__(timeout=None)
        self.entry = entry
        self.all_entries = all_entries
        self.list_page = list_page
        self.message: discord.Message | None = None

    @ui.button(label="Edit Topic", style=discord.ButtonStyle.primary)
    async def edit_topic(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        await interaction.response.send_modal(EditTopicModal(self.entry, self))

    @ui.button(label="Edit Summary", style=discord.ButtonStyle.primary)
    async def edit_summary(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        await interaction.response.send_modal(EditSummaryModal(self.entry, self))

    @ui.button(label="Delete", style=discord.ButtonStyle.danger)
    async def delete(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        await interaction.response.defer()
        db_delete_entry(self.entry["id"])
        remaining = [e for e in self.all_entries if e["id"] != self.entry["id"]]
        total_pages = max(1, -(-len(remaining) // PAGE_SIZE)) if remaining else 1
        new_page = min(self.list_page, total_pages - 1)
        new_view = HistoryListView(remaining, page=new_page)
        new_view.message = self.message
        embed = _build_list_embed(remaining, new_page, total_pages)
        try:
            await self.message.edit(embed=embed, view=new_view)
        except Exception as e:
            log.error("[review] delete back-to-list edit failed: %s", e)

    @ui.button(label="Back to list", style=discord.ButtonStyle.secondary)
    async def back(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        await interaction.response.defer()
        fresh = db_fetch_all()
        total_pages = max(1, -(-len(fresh) // PAGE_SIZE))
        new_page = min(self.list_page, total_pages - 1)
        new_view = HistoryListView(fresh, page=new_page)
        new_view.message = self.message
        embed = _build_list_embed(fresh, new_page, total_pages)
        try:
            await self.message.edit(embed=embed, view=new_view)
        except Exception as e:
            log.error("[review] back-to-list edit failed: %s", e)




async def start_history_review(interaction: discord.Interaction):
    entries = db_fetch_all()
    if not entries:
        await interaction.response.send_message("No knowledge entries found.", ephemeral=True)
        return
    total_pages = max(1, -(-len(entries) // PAGE_SIZE))
    view = HistoryListView(entries, page=0)
    embed = _build_list_embed(entries, 0, total_pages)
    channel = interaction.client.get_channel(SUPPORT_CHANNEL_ID)
    msg = await channel.send(embed=embed, view=view)
    view.message = msg
    await interaction.response.send_message("Review started.", ephemeral=True, delete_after=3)


# ══════════════════════════════════════════════════════════════════════════════
# RULES REVIEW
# ══════════════════════════════════════════════════════════════════════════════

def _build_rules_embed(section: dict, index: int, total: int) -> discord.Embed:
    embed = discord.Embed(
        title=f"Rules ({index + 1} of {total})",
        color=discord.Color.gold(),
    )
    embed.add_field(name=f"### {section['title']}", value=_truncate(section["body"], 900), inline=False)
    if section["subsections"]:
        subs = ", ".join(f"**{s['title']}**" for s in section["subsections"])
        embed.add_field(name="Subsections", value=subs, inline=False)
    return embed


class EditSectionModal(ui.Modal, title="Edit Section"):
    def __init__(self, section: dict, subsection_title: str | None, view: "RulesReviewView"):
        super().__init__()
        self._section = section
        self._subsection_title = subsection_title
        self._view = view

        if subsection_title:
            sub = next((s for s in section["subsections"] if s["title"] == subsection_title), None)
            default = sub["body"] if sub else ""
            label = f"**{subsection_title}**"
        else:
            default = section["body"]
            label = "Section body"

        self.content_input = ui.TextInput(
            label=label[:45],
            default=_truncate(default, 3900),
            max_length=4000,
            style=discord.TextStyle.paragraph,
        )
        self.add_item(self.content_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        cleaned = await llm_clean(self.content_input.value)
        if self._subsection_title:
            replace_subsection(self._section["title"], self._subsection_title, cleaned)
        else:
            replace_section(self._section["title"], cleaned)
        sections = parse_sections(load_rules())
        updated = next(
            (s for s in sections if s["title"].lower() == self._section["title"].lower()),
            self._section,
        )
        new_view = RulesReviewView(updated, self._view.index, len(sections))
        new_view.message = self._view.message
        embed = _build_rules_embed(updated, self._view.index, len(sections))
        try:
            await self._view.message.edit(embed=embed, view=new_view)
            log.info("[review] section '%s' updated", self._section["title"])
        except Exception as e:
            log.error("[review] EditSectionModal message.edit failed: %s", e)


class AddRuleModal(ui.Modal, title="Add New Rule"):
    def __init__(self, view: "RulesReviewView"):
        super().__init__()
        self._view = view
        self.title_input = ui.TextInput(
            label="Rule title (becomes ### heading)",
            max_length=200,
            style=discord.TextStyle.short,
        )
        self.body_input = ui.TextInput(
            label="Rule content",
            max_length=4000,
            style=discord.TextStyle.paragraph,
        )
        self.add_item(self.title_input)
        self.add_item(self.body_input)

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        cleaned_body = await llm_clean(self.body_input.value)
        add_section(self.title_input.value.strip(), cleaned_body)
        sections = parse_sections(load_rules())
        new_index = len(sections) - 1
        new_view = RulesReviewView(sections[new_index], new_index, len(sections))
        new_view.message = self._view.message
        embed = _build_rules_embed(sections[new_index], new_index, len(sections))
        try:
            await self._view.message.edit(embed=embed, view=new_view)
        except Exception as e:
            log.error("[review] AddRuleModal message.edit failed: %s", e)


class SectionNavSelect(ui.Select):
    """Navigate between ### sections via dropdown."""
    def __init__(self, sections: list[dict], current_index: int, view: "RulesReviewView"):
        self._sections = sections
        self._parent_view = view
        options = []
        for i, s in enumerate(sections[:25]):
            options.append(discord.SelectOption(
                label=s["title"][:100],
                value=str(i),
                default=(i == current_index),
            ))
        super().__init__(placeholder="Jump to section...", options=options, row=0)

    async def callback(self, interaction: discord.Interaction):
        if not await _check_admin(interaction):
            return
        await interaction.response.defer()
        index = int(self.values[0])
        section = self._sections[index]
        new_view = RulesReviewView(section, index, len(self._sections))
        new_view.message = self._parent_view.message
        embed = _build_rules_embed(section, index, len(self._sections))
        try:
            await self._parent_view.message.edit(embed=embed, view=new_view)
        except Exception as e:
            log.error("[review] SectionNavSelect edit failed: %s", e)


class EditSelect(ui.Select):
    """Edit whole section or a specific subsection."""
    def __init__(self, section: dict, view: "RulesReviewView"):
        self._section = section
        self._parent_view = view
        options = [discord.SelectOption(label="Whole section", value="__whole__")]
        for sub in section["subsections"]:
            options.append(discord.SelectOption(label=sub["title"], value=sub["title"]))
        super().__init__(placeholder="Select what to edit...", options=options[:25], row=1)

    async def callback(self, interaction: discord.Interaction):
        if not await _check_admin(interaction):
            return
        chosen = self.values[0]
        subsection = None if chosen == "__whole__" else chosen
        await interaction.response.send_modal(
            EditSectionModal(self._section, subsection, self._parent_view)
        )


class RulesReviewView(ui.View):
    def __init__(self, section: dict, index: int, total: int):
        super().__init__(timeout=None)
        self.section = section
        self.index = index
        self.total = total
        self.message: discord.Message | None = None

        sections = parse_sections(load_rules())
        self.add_item(SectionNavSelect(sections, index, self))
        self.add_item(EditSelect(section, self))

    @ui.button(label="Add New Rule", style=discord.ButtonStyle.secondary, row=2)
    async def add(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        await interaction.response.send_modal(AddRuleModal(self))

    @ui.button(label="Delete", style=discord.ButtonStyle.danger, row=2)
    async def delete(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        await interaction.response.defer()
        delete_section(self.section["title"])
        sections = parse_sections(load_rules())
        if not sections:
            try:
                await self.message.edit(content="No sections in rules.md.", embed=None, view=None)
            except Exception as e:
                log.error("[review] delete last section edit failed: %s", e)
            return
        index = max(0, min(self.index, len(sections) - 1))
        section = sections[index]
        new_view = RulesReviewView(section, index, len(sections))
        new_view.message = self.message
        embed = _build_rules_embed(section, index, len(sections))
        try:
            await self.message.edit(embed=embed, view=new_view)
        except Exception as e:
            log.error("[review] delete section edit failed: %s", e)

    @ui.button(label="✖ Close", style=discord.ButtonStyle.danger, row=2)
    async def close(self, interaction: discord.Interaction, button: ui.Button):
        if not await _check_admin(interaction):
            return
        await interaction.response.defer()
        try:
            await self.message.delete()
        except Exception as e:
            log.error("[review] RulesReviewView close failed: %s", e)


async def start_rules_review(interaction: discord.Interaction):
    sections = parse_sections(load_rules())
    if not sections:
        await interaction.response.send_message("rules.md has no ### sections yet.", ephemeral=True)
        return
    section = sections[0]
    view = RulesReviewView(section, index=0, total=len(sections))
    embed = _build_rules_embed(section, 0, len(sections))
    channel = interaction.client.get_channel(SUPPORT_CHANNEL_ID)
    msg = await channel.send(embed=embed, view=view)
    view.message = msg
    await interaction.response.send_message("Rules editor started.", ephemeral=True, delete_after=3)


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE LOG — clickable entry ID buttons
# ══════════════════════════════════════════════════════════════════════════════

class EntryButton(ui.Button):
    def __init__(self, entry_id: int, index: int):
        super().__init__(
            label=f"#{entry_id}",
            style=discord.ButtonStyle.secondary,
            row=index // 5,
        )
        self._entry_id = entry_id

    async def callback(self, interaction: discord.Interaction):
        from support_bot.rules import db_fetch_entry
        entry = db_fetch_entry(self._entry_id)
        if not entry:
            await interaction.response.send_message(
                f"Entry #{self._entry_id} not found.", ephemeral=True
            )
            return
        embed = _build_detail_embed(entry)
        await interaction.response.send_message(embed=embed, ephemeral=True)


class KnowledgeLogView(ui.View):
    def __init__(self, entry_ids: list[int]):
        super().__init__(timeout=300)
        for i, entry_id in enumerate(entry_ids[:25]):
            self.add_item(EntryButton(entry_id, i))