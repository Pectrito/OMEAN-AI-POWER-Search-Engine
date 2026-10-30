const form = document.querySelector("#search-form");
const input = document.querySelector("#query");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const answer = document.querySelector("#answer");

fetch("/api/health")
  .then((response) => {
    if (!response.ok) throw new Error(`Health check failed (${response.status}).`);
    return response.json();
  })
  .then((health) => {
    const model = health.neural_model;
    if (model?.trained) {
      status.textContent = `Searching ${health.indexed_documents.toLocaleString()} documents · trained ${model.parameters.toLocaleString()}-parameter model.`;
    } else if (model?.error) {
      status.textContent = `Searching ${health.indexed_documents.toLocaleString()} documents. ${model.error} The search app is still available.`;
    } else {
      status.textContent = `Searching ${health.indexed_documents.toLocaleString()} documents.`;
    }
  })
  .catch((error) => {
    status.textContent = error.message;
  });

function showResults(items) {
  results.replaceChildren();
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "No matching documents. Add text, Markdown, HTML, CSV, JSON, or RST files to the data folder and refresh.";
    results.append(empty);
    return;
  }

  for (const item of items) {
    const article = document.createElement("article");
    article.className = "result";
    const source = document.createElement("p");
    source.className = "result-source";
    const sourceLabel = item.source === "wikipedia"
      ? "Wikipedia article"
      : item.source === "web"
        ? "Collected page"
        : "Local file";
    source.textContent = `${sourceLabel} · ${item.path}`;
    const title = document.createElement("h2");
    title.textContent = item.title;
    const summary = document.createElement("p");
    summary.className = "result-summary";
    summary.textContent = item.summary;
    const feedback = document.createElement("div");
    feedback.className = "feedback";
    feedback.dataset.sourceId = item.source_id;
    for (const [label, relevant] of [["Helpful", true], ["Not relevant", false]]) {
      const button = document.createElement("button");
      button.className = "feedback-button";
      button.type = "button";
      button.textContent = label;
      button.dataset.relevant = String(relevant);
      feedback.append(button);
    }
    article.append(source, title, summary, feedback);
    results.append(article);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  status.textContent = "Searching…";
  results.replaceChildren();
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    if (!response.ok) throw new Error(`Search failed (${response.status}).`);
    const data = await response.json();
    showResults(data.results);
    status.textContent = `${data.results.length} result${data.results.length === 1 ? "" : "s"} for “${data.query}”`;
    answer.replaceChildren();
    if (/^(what|who|when|where|why|how|which|define)\b/i.test(query) && data.results.length) {
      const heading = document.createElement("h2");
      heading.textContent = "Quick answer";
      const content = document.createElement("p");
      content.textContent = data.results[0].summary;
      answer.append(heading, content);
      answer.hidden = false;
    } else {
      answer.hidden = true;
    }
  } catch (error) {
    status.textContent = error.message;
  }
});

results.addEventListener("click", async (event) => {
  const button = event.target.closest(".feedback-button");
  if (!button) return;
  button.disabled = true;
  try {
    const response = await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: input.value.trim(),
        source_id: button.parentElement.dataset.sourceId,
        relevant: button.dataset.relevant === "true",
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Feedback failed (${response.status}).`);
    status.textContent = data.trained
      ? `Learned from ${data.feedback_examples} relevance labels.`
      : `Saved feedback (${data.feedback_examples}); add more positive and negative examples to train ranking.`;
    button.textContent = "Saved";
  } catch (error) {
    button.disabled = false;
    status.textContent = error.message;
  }
});

document.querySelector("#reindex").addEventListener("click", async () => {
  status.textContent = "Refreshing index…";
  try {
    const response = await fetch("/api/reindex", { method: "POST" });
    if (!response.ok) throw new Error(`Index refresh failed (${response.status}).`);
    const data = await response.json();
    status.textContent = `Indexed ${data.indexed} local files; ${data.indexed_documents.toLocaleString()} documents searchable. Removed ${data.removed} stale entries.`;
    if (data.errors.length) status.textContent += ` ${data.errors.length} file(s) could not be indexed.`;
  } catch (error) {
    status.textContent = error.message;
  }
});
