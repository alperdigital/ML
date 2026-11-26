(() => {
  const layerControl = document.querySelector('.layer-toggle');
  const dataLayer = document.querySelector('.data-layer');
  layerControl.hidden = false;
  layerControl.addEventListener('click', () => {
    const hidden = dataLayer.toggleAttribute('hidden');
    document.querySelector('.layer-description').hidden = hidden;
    layerControl.setAttribute('aria-pressed', String(!hidden));
    layerControl.querySelector('span').textContent = hidden ? '＋' : '−';
  });
  const versions = Array.from(document.querySelectorAll('.stage-link'));
  const panels = versions.map((link) => document.getElementById(link.hash.slice(1)));
  const menu = document.querySelector('.version-menu');
  menu.setAttribute('role', 'tablist');
  const compact = window.matchMedia('(max-width: 680px)');
  function setOrientation() {
    menu.setAttribute('aria-orientation', compact.matches ? 'horizontal' : 'vertical');
  }
  setOrientation();
  compact.addEventListener('change', setOrientation);
  function select(index) {
    versions.forEach((link, i) => {
      const active = i === index;
      link.setAttribute('aria-selected', String(active));
      link.classList.toggle('selected', active);
      link.tabIndex = active ? 0 : -1;
      panels[i].hidden = !active;
    });
  }
  versions.forEach((link, index) => {
    link.id = 'version-' + link.dataset.version;
    link.setAttribute('role', 'tab');
    link.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].setAttribute('aria-labelledby', link.id);
    panels[index].tabIndex = 0;
    link.addEventListener('click', (event) => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      select(index);
    });
    link.addEventListener('keydown', (event) => {
      const keys = compact.matches ? ['ArrowLeft', 'ArrowRight'] : ['ArrowUp', 'ArrowDown'];
      let next;
      if (event.key === keys[0]) next = (index + versions.length - 1) % versions.length;
      else if (event.key === keys[1]) next = (index + 1) % versions.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = versions.length - 1;
      else if (event.key === ' ') next = index;
      else return;
      event.preventDefault();
      select(next);
      versions[next].focus();
    });
  });
  function selectHash() {
    const index = versions.findIndex((link) => link.hash === location.hash);
    if (index >= 0) select(index);
    return index >= 0;
  }
  if (!selectHash()) select(0);
  window.addEventListener('hashchange', selectHash);
  const readers = document.querySelectorAll('[name=reading]');
  function setReading() {
    document.documentElement.dataset.reading = document.querySelector('[name=reading]:checked').value;
  }
  readers.forEach((input) => input.addEventListener('change', setReading));
  setReading();
})();
