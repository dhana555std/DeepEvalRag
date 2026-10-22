/*
 * report.js - theme switcher (Light / Dark) and client-side pagination of the case details.
 *
 * The report page contains EVERY case (<details data-case> inside #case-list). This script shows
 * `data-page-size` cases at a time (default 20) and builds the pager (<nav data-pager>) at the top
 * and bottom of the list. Without JavaScript, or when printing, all cases are simply visible.
 *
 * Pager: "‹ Prev" / "Next ›" step one page; page numbers are shown in sheets of at most
 * `data-pager-pages` (default 20) numbers - "«" / "»" jump to the previous / next sheet and only
 * appear when another sheet exists.
 *
 * Filters (above the list): state buttons [data-filter] (all / fail / err / pass), a category <select
 * data-category> and a search box [data-search]. Pagination then runs over the matching cases only.
 * Each case carries data-state, data-category and data-text (lower-cased id + question).
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

  /** Theme switcher: Dark (default) / Light, remembered in localStorage. */
  function initTheme() {
    var root = document.documentElement;
    var buttons = Array.prototype.slice.call(document.querySelectorAll('[data-theme-set]'));
    if (!buttons.length) return;
    function apply(mode, save) {
      root.setAttribute('data-theme', mode);
      buttons.forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-theme-set') === mode)); });
      if (!save) return;
      try {
        localStorage.setItem('ragqa-theme', mode);
      } catch (e) { /* file:// or private mode: the choice just is not remembered */ }
    }
    var saved = null;
    try { saved = localStorage.getItem('ragqa-theme'); } catch (e) { /* ignore */ }
    apply(saved === 'light' || saved === 'dark' ? saved : (root.getAttribute('data-default-theme') || 'dark'), false);
    buttons.forEach(function (b) { b.addEventListener('click', function () { apply(b.getAttribute('data-theme-set'), true); }); });
  }

  /** Tabs: Overview (default) / Results by layer / Case details. Case links and pager hashes open Case details. */
  function initTabs() {
    var tabs = Array.prototype.slice.call(document.querySelectorAll('[data-tab]'));
    if (!tabs.length) return;
    function select(name, focus) {
      tabs.forEach(function (t) {
        var on = t.getAttribute('data-tab') === name;
        t.setAttribute('aria-selected', String(on));
        t.tabIndex = on ? 0 : -1;
        if (on && focus) t.focus();
      });
      Array.prototype.forEach.call(document.querySelectorAll('[data-panel]'), function (p) {
        p.hidden = p.getAttribute('data-panel') !== name;
      });
    }
    function fromHash() {
      var h = location.hash || '';
      if (h.indexOf('#case-') === 0 || h.indexOf('#page=') === 0) select('cases', false);
    }
    tabs.forEach(function (t, i) {
      t.addEventListener('click', function () { select(t.getAttribute('data-tab'), false); });
      t.addEventListener('keydown', function (e) {
        var d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
        if (d) { e.preventDefault(); select(tabs[(i + d + tabs.length) % tabs.length].getAttribute('data-tab'), true); }
      });
    });
    window.addEventListener('hashchange', fromHash);
    fromHash();
  }

  function init() {
    initTabs();
    initTheme();
    var list = document.getElementById('case-list');
    if (!list) return;
    var cases = Array.prototype.slice.call(list.querySelectorAll('[data-case]'));
    var pageSize = parseInt(list.getAttribute('data-page-size'), 10) || 0;
    var sheet = parseInt(list.getAttribute('data-pager-pages'), 10) || 20;
    var filters = {
      state: 'all', category: '', text: ''
    };
    var matching = cases;
    var total = 1;
    var filterBar = document.querySelector('.case-filter');
    var emptyMsg = document.createElement('p');
    emptyMsg.className = 'no-cases';
    emptyMsg.textContent = 'No cases match the current filter.';
    emptyMsg.hidden = true;
    list.appendChild(emptyMsg);

    function recount() {
      matching = cases.filter(function (el) {
        return (filters.state === 'all' || el.getAttribute('data-state') === filters.state) &&
          (!filters.category || el.getAttribute('data-category') === filters.category) &&
          (!filters.text || (el.getAttribute('data-text') || '').indexOf(filters.text) >= 0);
      });
      total = pageSize > 0 ? Math.max(1, Math.ceil(matching.length / pageSize)) : 1;
    }
    recount();
    // nothing to filter or page through: leave every case visible and skip the controls' logic
    if (total <= 1 && !filterBar) return;

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
      var size = pageSize > 0 ? pageSize : matching.length;
      var from = (current - 1) * size;
      var to = Math.min(matching.length, from + size);
      var visible = matching.slice(from, to);
      cases.forEach(function (el) { el.hidden = visible.indexOf(el) < 0; });
      emptyMsg.hidden = matching.length > 0;
      if (info) {
        var filtered = matching.length !== cases.length;
        info.textContent = matching.length
          ? 'Showing cases ' + (from + 1) + '–' + to + ' of ' + matching.length + (filtered ? ' (filtered from ' + cases.length + ')' : '') +
            (!filtered && info.getAttribute('data-suffix') ? ' ' + info.getAttribute('data-suffix') : '')
          : '';
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
        var target = cases.filter(function (el) { return '#' + el.id === h; })[0];
        if (target) {
          if (matching.indexOf(target) < 0) resetFilters();
          var idx = matching.indexOf(target);
          show(pageSize > 0 ? Math.floor(idx / pageSize) + 1 : 1, false);
          target.open = true;
          target.scrollIntoView();
          return;
        }
      }
      show(1, false);
    }

    function resetFilters() {
      filters.state = 'all'; filters.category = ''; filters.text = '';
      Array.prototype.forEach.call(document.querySelectorAll('[data-filter]'), function (b) {
        b.setAttribute('aria-pressed', String(b.getAttribute('data-filter') === 'all'));
      });
      var sel = document.querySelector('[data-category]'); if (sel) sel.value = '';
      var box = document.querySelector('[data-search]'); if (box) box.value = '';
      recount();
    }

    function refilter() { recount(); show(1, false); }

    Array.prototype.forEach.call(document.querySelectorAll('[data-filter]'), function (b) {
      b.addEventListener('click', function () {
        filters.state = b.getAttribute('data-filter');
        Array.prototype.forEach.call(document.querySelectorAll('[data-filter]'), function (o) {
          o.setAttribute('aria-pressed', String(o === b));
        });
        refilter();
      });
    });
    var catSel = document.querySelector('[data-category]');
    if (catSel) catSel.addEventListener('change', function () { filters.category = catSel.value; refilter(); });
    var searchBox = document.querySelector('[data-search]');
    if (searchBox) searchBox.addEventListener('input', function () { filters.text = searchBox.value.trim().toLowerCase(); refilter(); });

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
