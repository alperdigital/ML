/* A read-only view of the same holdout rows used in the portfolio's comparison. */
(() => {
  const currency = new Intl.NumberFormat("tr-TR", {minimumFractionDigits: 2, maximumFractionDigits: 2});
  const price = (v) => currency.format(v) + " $";
  const template = document.createElement("template");
  template.innerHTML = `
    <style>
      :host{display:block;box-sizing:border-box;width:min(100%,1120px);margin:clamp(2rem,6vw,5rem) auto;color:#222629;--accent:#214bad;font:inherit}
      *,*:before,*:after{box-sizing:border-box}
      section{padding:clamp(1.3rem,4.6vw,3.5rem);border:1px solid #d3d7dd;border-radius:1.3rem;background:#f3f5f7;text-align:left}
      .intro{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(13rem,.7fr);gap:2rem;align-items:end}
      h3{font:inherit;font-size:clamp(1.6rem,3.35vw,2.55rem);font-weight:600;letter-spacing:-.055em;line-height:1.07;margin:0}
      p{font-size:1rem;line-height:1.5;margin:.7rem 0 0;color:#4b525a;max-width:38em}
      label{font-size:.875rem;display:block;color:#3b434c;margin-bottom:.35rem}
      select{width:100%;font:inherit;font-size:1rem;font-weight:550;color:#20304a;background:#fff;border:1px solid #a1a9b5;border-radius:.6rem;padding:.7rem 2.15rem .7rem .8rem;cursor:pointer}
      select:focus-visible,a:focus-visible{outline:3px solid #2853b4;outline-offset:3px}
      .sale{display:flex;align-items:baseline;justify-content:space-between;gap:1rem;border-bottom:1px solid #c4cad1;padding:1.5rem 0 .75rem;margin-top:.7rem}
      .sale span{font-size:1rem}.sale strong{font-size:clamp(1.4rem,3vw,2rem);letter-spacing:-.045em;font-variant-numeric:tabular-nums}
      .estimates{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1.4rem;margin:1.1rem 0}
      .estimates span,.estimates small{display:block;color:#4b525a;font-size:.9rem}
      .estimates strong{display:block;margin:.25rem 0;font-variant-numeric:tabular-nums;font-size:clamp(1.2rem,2.7vw,1.85rem);letter-spacing:-.045em;line-height:1.15}
      .estimates div:last-child strong{color:var(--accent)}
      .source{display:inline-block;font:inherit;color:#394963;text-decoration:underline;text-underline-offset:.2em;font-size:.875rem;margin-top:.35rem}
      details{border-top:1px solid #c4cad1;padding-top:.8rem;margin-top:.7rem}
      summary{cursor:pointer;font-size:.9rem;font-weight:600;color:#25364d}
      code{display:block;background:#fff;border:1px solid #d3d7dd;border-radius:.45rem;padding:.7rem;margin:.65rem 0;font-size:.85rem;overflow-wrap:anywhere}
      [hidden]{display:none!important}
      @media(max-width:680px){.intro{grid-template-columns:1fr;gap:1rem}.estimates{grid-template-columns:1fr;gap:0;margin:.2rem 0 .45rem}.estimates div{display:grid;grid-template-columns:2.3em 1fr;align-items:baseline;gap:.15rem .55rem;padding:.7rem 0;border-bottom:1px solid #d5dae1}.estimates strong{font-size:1.4rem;margin:0}.estimates small{grid-column:2}.sale{margin-top:.15rem;padding-top:1rem}}
    </style>
    <section aria-labelledby="case-title">
      <div class="intro">
        <div><h3 id="case-title">Bir evde üç tahmin</h3>
        <p class="closest">Bir kayıt seçerek üç tahminin gerçek satış fiyatından ne kadar uzaklaştığını görebilirsin.</p></div>
        <div><label for="house">Satış kaydı seç</label><select id="house" disabled><option>Kayıtlar yükleniyor…</option></select></div>
      </div>
      <details hidden><summary>Bu evin V2 tahminini yeniden üret</summary>
        <p>Proje klasöründe bu komutu çalıştır. Yayımlanan CatBoost modeli yalnızca eğitim evlerinden öğrenir ve seçtiğin evi satış fiyatını görmeden değerlendirir.</p>
        <code></code>
      </details>
      <div class="result" aria-live="polite" aria-atomic="true" hidden>
        <div class="sale"><span>Gerçekleşen satış</span><strong></strong></div>
        <div class="estimates">
          <div><span>V1</span><strong></strong><small></small></div>
          <div><span>V2</span><strong></strong><small></small></div>
          <div><span>V3</span><strong></strong><small></small></div>
        </div>
      </div>
      <a class="source" href="https://github.com/alperdigital/ML/blob/v3/ames-v3/results/three-version-holdout.csv">292 evin tahminlerini incele</a>
    </section>`;
  class AmesCaseStudy extends HTMLElement {
    connectedCallback() {
      if (this.shadowRoot) return;
      const view = this.attachShadow({mode: "open"});
      view.append(template.content.cloneNode(true));
      const select = view.querySelector("select");
      fetch(new URL("./house-results.json", document.baseURI))
        .then((response) => {
          if (!response.ok) throw new Error("unavailable");
          return response.json();
        }).then((rows) => {
          select.replaceChildren();
          const closest = [0, 0, 0];
          for (const [id, zone, year, actual, ...estimates] of rows) {
            select.add(new Option("#" + id + " · " + zone + " · " + year, String(id)));
            const errors = estimates.map(estimate => Math.abs(Math.round(estimate * 100) - Math.round(actual * 100)));
            const smallest = Math.min(...errors);
            if (errors.filter(error => error === smallest).length === 1) closest[errors.indexOf(smallest)]++;
          }
          const ties = rows.length - closest.reduce((a, b) => a + b, 0);
          view.querySelector(".closest").textContent = `Gerçek fiyata ${closest[0]} evde V1, ${closest[1]} evde V2, ${closest[2]} evde V3 en çok yaklaştı.` +
            (ties ? ` ${ties} evde birincilik paylaşıldı.` : "") + " Bir kayıt seçip üç tahmini karşılaştırabilirsin.";
          select.value = "899";
          const render = () => {
            const row = rows[select.selectedIndex];
            const actual = row[3];
            view.querySelector("code").textContent = "python ames-v3/predict.py --version v2 --holdout-id " + row[0];
            view.querySelector(".sale strong").textContent = price(actual);
            view.querySelectorAll(".estimates > div").forEach((el,i) => {
              const estimate = row[4+i];
              const delta = Math.round(estimate * 100) - Math.round(actual * 100);
              el.querySelector("strong").textContent = price(estimate);
              el.querySelector("small").textContent = delta === 0 ? "Satış fiyatıyla aynı" :
                price(Math.abs(delta) / 100) + (delta > 0 ? " üzerinde" : " altında");
            });
          };
          select.addEventListener("change", render);
          render();
          select.disabled = false;
          view.querySelector(".result").hidden = false;
          view.querySelector("details").hidden = false;
        }).catch(() => { select.replaceChildren(new Option("Kayıtlar yüklenemedi")); });
    }
  }
  customElements.define("ames-case-study", AmesCaseStudy);
})();
