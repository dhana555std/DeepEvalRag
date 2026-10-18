/*
 * report.js - theme switcher (System / Light / Dark) and client-side pagination of the case details.
 *
 * The report page contains EVERY case (<details data-case> inside #case-list). This script shows
 * `data-page-size` cases at a time (default 20) and builds the pager (<nav data-pager>) at the top
 * and bottom of the list. Without JavaScript, or when printing, all cases are simply visible.
 *
 * Pager: "‹ Prev" / "Next ›" step one page; page numbers are shown in sheets of at most
 * `data-pager-pages` (default 20) numbers - "«" / "»" jump to the previous / next sheet and only
 * appear when another sheet exists.
 *
 * State lives in the URL hash so a page can be bookmarked or shared and survives a refresh:
 *   #page=3        show page 3
 *   #case-Q-17     show the page containing case Q-17 and open it
 *
 * Markup contract (see templates/report.html.j2):
 *   #case-list[data-page-size][data-pager-pages]   container of all cases
 *   [data-case]                                    one case (a <details> element)
 *   [data-pager]                                   pager placeholder(s) filled by this script
 *   [data-page-info][data-suffix]                  "Showing cases a-b of n" line
 * Brand-neutral: all styling is in report.css (.pager, .pager .current). No dependencies.
 */
(function () {
  'use strict';

  /** Pager model for `current` of `total` pages; `size` = max page numbers shown at once. */
  function buildPager(current, total, size) {
    if (total <= 1) return [];
    size = Math.max(1, size || 20);
    var start = Math.floor((current - 1) / size) * size + 1;
    var end = Math.min(total, start + size - 1);
    var items = [];
    if (start > 1) {
      items.push({ label: '«', target: start - size, title: 'Pages ' + (start - size) + '-' + (start - 1) });
    }
    items.push({ label: '‹ Prev', target: current > 1 ? current - 1 : null, title: 'Previous page' });
    for (var n = start; n <= end; n++) {
      items.push({ label: String(n), target: n, current: n === current });
    }
    items.push({ label: 'Next ›', target: current < total ? current + 1 : null, title: 'Next page' });
    if (end < total) {
      items.push({ label: '»', target: end + 1, title: 'Pages ' + (end + 1) + '-' + Math.min(total, end + size) });
    }
    return items;
  }

  /** Theme switcher: System (default) / Light / Dark, remembered in localStorage. */
  function initTheme() {
    var root = document.documentElement;
    var buttons = Array.prototype.slice.call(document.querySelectorAll('[data-theme-set]'));
    if (!buttons.length) return;
    function apply(mode, save) {
      root.setAttribute('data-theme', mode);
      buttons.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-theme-set') === mode)); });
      if (!save) return;
      try {
        if (mode === 'system') localStorage.removeItem('ragqa-theme'); else localStorage.setItem('ragqa-theme', mode);
      } catch (e) { /* file:// or private mode: the choice just is not remembered */ }
    }
    var saved = null;
    try { saved = localStorage.getItem('ragqa-theme'); } catch (e) { /* ignore */ }
    apply(saved === 'light' || saved === 'dark' ? saved : 'system', false);
    buttons.forEach(function (b) { b.addEventListener('click', function () { apply(b.getAttribute('data-theme-set'), true); }); });
  }

  function init() {
    initTheme();
    var list = document.getElementById('case-list');
    if (!list) return;
    var cases = Array.prototype.slice.call(list.querySelectorAll('[data-case]'));
    var pageSize = parseInt(list.getAttribute('data-page-size'), 10) || 0;
    var sheet = parseInt(list.getAttribute('data-pager-pages'), 10) || 20;
    var total = pageSize > 0 ? Math.ceil(cases.length / pageSize) : 1;
    if (total <= 1) return; // everything fits on one page: leave all cases visible, no pager

    var navs = Array.prototype.slice.call(document.querySelectorAll('[data-pager]'));
    var info = document.querySelector('[data-page-info]');
    var current = 1;

    function renderPager() {
      var items = buildPager(current, total, sheet);
      navs.forEach(function (nav) {
        nav.textContent = '';
        items.forEach(function (it) {
          var b = document.createElement('button');
          b.type = 'button';
          b.textContent = it.label;
          if (it.title) b.title = it.title;
          if (it.current) { b.className = 'current'; b.setAttribute('aria-current', 'page'); }
          if (it.target === null) {
            b.disabled = true;
          } else if (!it.current) {
            b.addEventListener('click', function () { show(it.target, true); });
          }
          nav.appendChild(b);
        });
        nav.hidden = false;
      });
    }

    function show(page, userAction) {
      current = Math.min(Math.max(1, page), total);
      var from = (current - 1) * pageSize;
      var to = Math.min(cases.length, from + pageSize);
      cases.forEach(function (el, i) { el.hidden = i < from || i >= to; });
      if (info) {
        info.textContent = 'Showing cases ' + (from + 1) + '–' + to + ' of ' + cases.length +
          (info.getAttribute('data-suffix') ? ' ' + info.getAttribute('data-suffix') : '');
      }
      renderPager();
      try { history.replaceState(null, '', '#page=' + current); } catch (e) { /* file:// restrictions */ }
      if (userAction) {
        var anchor = document.getElementById('cases');
        if (anchor && anchor.scrollIntoView) anchor.scrollIntoView();
      }
    }

    function fromHash() {
      var h = decodeURIComponent(location.hash || '');
      var m = /^#page=(\d+)$/.exec(h);
      if (m) return show(parseInt(m[1], 10), false);
      if (h.indexOf('#case-') === 0) {
        var idx = cases.findIndex(function (el) { return '#' + el.id === h; });
        if (idx >= 0) {
          show(Math.floor(idx / pageSize) + 1, false);
          cases[idx].open = true;
          cases[idx].scrollIntoView();
          return;
        }
      }
      show(1, false);
    }

    window.addEventListener('hashchange', fromHash);
    fromHash();
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { buildPager: buildPager }; // for the unit tests (node)
  }
  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
  }
})();
