// "Compare with" menus (research_tools.tables.show_tabs): in every tab of the group, show the version of the
// table compared with the chosen tab, or the plain table in the chosen tab itself.
document.addEventListener('change', (event) => {
  const menu = event.target.closest('.compare-control select');
  if (!menu) return;
  menu.closest('.compare-group').querySelectorAll('.tab-pane').forEach((pane) => {
    const versions = [...pane.querySelectorAll('.cmp-variant')];
    const shown = versions.some((v) => v.dataset.cmp === menu.value) ? menu.value : 'none';
    versions.forEach((v) => { v.hidden = v.dataset.cmp !== shown; });
  });
});
