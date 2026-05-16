const root = document.documentElement;
const sidebar = document.getElementById("sidebar");
const mainWrapper = document.getElementById("mainWrapper");
const sidebarToggle = document.getElementById("sidebarToggle");
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");

function setTheme(theme) {
  root.setAttribute("data-theme", theme);
  localStorage.setItem("studentai-theme", theme);
  if (themeIcon) {
    themeIcon.className = theme === "light" ? "bi bi-sun-fill" : "bi bi-moon-fill";
  }
}

setTheme(localStorage.getItem("studentai-theme") || "dark");

sidebarToggle?.addEventListener("click", () => {
  if (window.innerWidth <= 768) {
    sidebar?.classList.toggle("open");
  } else {
    sidebar?.classList.toggle("collapsed");
    mainWrapper?.classList.toggle("full");
  }
});

themeToggle?.addEventListener("click", () => {
  const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
  setTheme(next);
});

document.addEventListener("click", (event) => {
  if (window.innerWidth > 768 || !sidebar?.classList.contains("open")) return;
  const target = event.target;
  if (!sidebar.contains(target) && !sidebarToggle?.contains(target)) {
    sidebar.classList.remove("open");
  }
});
