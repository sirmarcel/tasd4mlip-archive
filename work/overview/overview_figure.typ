// The preprint's overview figure. Data in output/build (overview_figure.py), styling here.
// Compiled by overview_figure.py; by hand: typst compile overview_figure.typ figures/overview.pdf
#set page(width: 5.5in, height: auto, margin: 3pt)
#set text(font: "Times New Roman", size: 8pt)

#let S = json("output/build/structure.json")
#let Sm = json("output/build/style.json")
#let T = json("output/build/toy.json")
#let K = S.K

// ---- palettes
#let hopcol(k) = rgb(Sm.hopcolors.at(k))                    // shared with overview_figure.py through style.json
#let starcol = ("#0077BB", "#EE7733", "#009988", "#EE3377", "#33BBEE", "#BBBBBB").map(rgb)
#let tint(h) = (100%, 60%, 30%).at(h)
#let ink = black
#let ghost = luma(222)
#let ghostbond = luma(232)
#let unused = luma(218)
#let alarm = rgb("#CC3311")

#let small(body) = text(6.5pt, body)
#let panel(l, title, note: none) = block(below: 3pt)[#text(weight: "bold", 8.5pt)[#l #h(1pt) #title]#if note != none [#h(6pt)#note]]

// ============================================================================
// (a) the mechanism on a toy chain
// ============================================================================
#let cellgrid(nrow, ncol, size, cell) = box(width: ncol * size, height: nrow * size, {
  for i in range(nrow) { for j in range(ncol) { place(dx: j * size, dy: i * size, cell(i, j)) } }
})
#let cell(size, fill: none, stroke: 0.25pt + luma(205)) = rect(width: size, height: size, fill: fill, stroke: stroke)
#let dotted(size, fill: none) = box(width: size, height: size, {
  place(cell(size, fill: fill))
  let r = 0.17 * size
  place(dx: size / 2 - r, dy: size / 2 - r, circle(radius: r, fill: black, stroke: 0.35pt + white))
})
#let crossed(size) = box(width: size, height: size, {
  place(cell(size))
  let m = 0.25 * size
  place(line(start: (m, m), end: (size - m, size - m), stroke: 0.5pt + luma(120)))
  place(line(start: (m, size - m), end: (size - m, m), stroke: 0.5pt + luma(120)))
})

#let chain(size, colof, bond: 1.1pt + luma(160), edge: 0.3pt + luma(90)) = {
  let n = T.n
  box(width: n * size, height: size, {
    for e in T.edges {
      place(line(start: ((e.at(0) + 0.5) * size, size / 2), end: ((e.at(1) + 0.5) * size, size / 2), stroke: bond))
    }
    for j in range(n) {
      let r = 0.36 * size
      place(dx: (j + 0.5) * size - r, dy: size / 2 - r, circle(radius: r, fill: colof(j), stroke: edge))
    }
  })
}

#let asdrow(tag, size) = {
  let R = T.at(tag)
  let n = T.n
  let c = R.num_colors
  let colof(j) = starcol.at(R.colors.at(j))
  let bad = R.contaminated.map(b => R.reads.find(r => r.at(0) == b.at(0) and r.at(1) == b.at(1)))
  let badcells = bad.map(r => (r.at(2), r.at(3)))
  let readfrom = (:)
  for r in R.reads { readfrom.insert(str(r.at(0)) + "," + str(r.at(1)), (r.at(2), r.at(3))) }
  let readcells = R.reads.map(r => (r.at(2), r.at(3)))
  let Hgrid = cellgrid(n, n, size, (i, j) => {
    let h = T.hop.at(i).at(j)
    if h < 0 { cell(size) }
    else if R.pattern.at(i).at(j) {
      let (e, k) = readfrom.at(str(i) + "," + str(j))
      cell(size, fill: starcol.at(k).transparentize(100% - tint(h)))   // color of the compressed cell it is read from
    }
    else { cell(size, fill: unused) }                                  // nonzero, outside the assumed pattern: discarded
  })
  let Sgrid = cellgrid(n, c, size, (j, k) => cell(size, fill: if R.colors.at(j) == k { starcol.at(k) } else { none }))
  let Cgrid = cellgrid(n, c, size, (i, k) => {
    let v = R.compressed.at(i).at(k)
    let isread = readcells.contains((i, k))
    if v != 0 and not isread { crossed(size) }                       // a sum of several entries, never read
    else {
      let f = if v == 0 { none } else { starcol.at(k).transparentize(100% - tint(calc.min(2, calc.floor(-calc.log(v, base: 10) + 0.3)))) }
      if badcells.contains((i, k)) { dotted(size, fill: f) } else { cell(size, fill: f) }
    }
  })
  // operators drawn as shapes, so they sit exactly at the matrix mid-height (glyph boxes do not)
  let H = n * size
  let opdot = box(width: 6pt, height: H, place(dx: 3pt - 0.75pt, dy: H / 2 - 0.75pt, circle(radius: 0.75pt, fill: black)))
  let opeq = box(width: 7pt, height: H, {
    place(line(start: (1pt, H / 2 - 1pt), end: (6pt, H / 2 - 1pt), stroke: 0.55pt + black))
    place(line(start: (1pt, H / 2 + 1pt), end: (6pt, H / 2 + 1pt), stroke: 0.55pt + black))
  })
  let head(s) = align(center, small(s))
  // chain over the columns of H, then H . S = HS with names above
  grid(columns: 5, column-gutter: 1pt, row-gutter: 2pt, align: (center + bottom),
    head[$upright(bold(H))$], [], head[$upright(bold(S))$], [], head[$upright(bold(H)) upright(bold(S))$],
    Hgrid, opdot, Sgrid, opeq, Cgrid,
    head[Hessian], [], head[seeds], [], head[compressed])
}

#let swatch(c) = box(cell(5pt, fill: c, stroke: 0.25pt + luma(160)))
#let keytext(body) = text(6.5pt, body)
#let key = {
  let c = starcol.at(0)
  let shade = box(grid(columns: 3, column-gutter: 1pt, ..(100%, 60%, 30%).map(t => cell(5pt, fill: c.transparentize(100% - t), stroke: 0.25pt + luma(160)))))
  block(fill: luma(247), radius: 3pt, inset: (x: 6pt, y: 5pt),
    stack(dir: ttb, spacing: 3.5pt,
      align(left, text(6.5pt, weight: "bold")[Legend]),
      grid(columns: (17pt, auto), column-gutter: 4pt, row-gutter: 3pt, align: (center + horizon, left + horizon),
        swatch(c), keytext[*hue:* one seed, one HVP, one column of $upright(bold(H)) upright(bold(S))$],
        [], keytext[*in $upright(bold(H))$:* the $upright(bold(H)) upright(bold(S))$ entry it is decompressed from],
        shade, keytext[*shade:* hop distance (0, 1, 2 hops)],
        swatch(unused), keytext[*gray:* outside the truncated pattern],
        box(crossed(5pt)), keytext[*crossed:* nonzero but unused in decompression],
        box(dotted(5pt)), keytext[*dot:* contaminated by a truncated entry])))
}

#let schematic(size: 5.8pt) = {
  let n = T.n
  let case(tag, title) = stack(dir: ttb, spacing: 4pt, align(center, small(title)), asdrow(tag, size))
  grid(columns: (auto, 1fr, auto, 1fr, auto), align: horizon,
    case("exact")[exact pattern, $k = K = #T.K$: #T.exact.num_colors HVPs],
    [],
    case("trunc")[truncated, $k = #T.k$: #T.trunc.num_colors HVPs],
    [],
    key)
}

// ============================================================================
// (b), (c) the real thing, locally: one atom, its hop shells, its Hessian row
// ============================================================================
#let structure(W, H, mode: "hop", view: S.view) = {
  let s = W / (2 * view)                       // pt per Angstrom
  let X(a) = W / 2 + a.x * s
  let Y(a) = H / 2 - a.y * s
  let byid = (:)
  for a in S.atoms { byid.insert(str(a.i), a) }
  let visible(a) = calc.abs(a.x) < view + 1 and calc.abs(a.y) < H / (2 * s) + 1
  box(width: W, height: H, clip: true, {
    for b in S.bonds {
      let (p, q) = (byid.at(str(b.i)), byid.at(str(b.j)))
      if not b.incluster and (visible(p) or visible(q)) {
        place(line(start: (X(p), Y(p)), end: (X(q), Y(q)), stroke: 0.5pt + ghostbond))
      }
    }
    for a in S.atoms {
      if a.hop < 0 and visible(a) {
        let r = 0.22 * a.r * s
        place(dx: X(a) - r, dy: Y(a) - r, circle(radius: r, fill: luma(242), stroke: 0.25pt + ghost))
      }
    }
    for b in S.bonds {
      let (p, q) = (byid.at(str(b.i)), byid.at(str(b.j)))
      if b.incluster { place(line(start: (X(p), Y(p)), end: (X(q), Y(q)), stroke: 1.0pt + luma(130))) }
    }
    let c = byid.at(str(S.centre))
    if mode == "hop" {
      let rc = S.cutoff_r * s
      place(dx: X(c) - rc, dy: Y(c) - rc, circle(radius: rc, fill: none, stroke: (paint: hopcol(1), thickness: 0.6pt, dash: (2pt, 1.5pt))))
      place(dx: X(c) - 1.1 * rc, dy: Y(c) + rc + 1pt, box(fill: white, inset: 1.5pt, text(7.5pt)[cutoff $r_"c"$]))
      for j in S.onehop {
        let q = byid.at(str(j))
        place(line(start: (X(c), Y(c)), end: (X(q), Y(q)), stroke: 0.6pt + hopcol(1)))
      }
    }
    for a in S.atoms {
      if a.hop > 0 {
        let r = 0.42 * a.r * s
        let fill = if mode == "hop" { hopcol(a.hop) } else { rgb(a.magcol) }
        place(dx: X(a) - r, dy: Y(a) - r, circle(radius: r, fill: fill, stroke: 0.3pt + white))
      }
    }
    let r = 0.42 * c.r * s + 0.5pt
    place(dx: X(c) - r, dy: Y(c) - r, circle(radius: r, fill: white, stroke: 1.1pt + black))
  })
}

#let hoplegend() = {
  let items = ()
  items.push(box(circle(radius: 2.2pt, fill: white, stroke: 0.8pt + black)))
  items.push(small[0])
  for k in range(1, K + 1) {
    items.push(box(circle(radius: 2.2pt, fill: hopcol(k), stroke: none)))
    items.push(small[#k])
  }
  grid(columns: 2 * (K + 1), column-gutter: 2.5pt, align: horizon, ..items)
}

#let colorbar(W, lo, hi) = stack(dir: ttb, spacing: 1.5pt,
  rect(width: W, height: 4pt, fill: gradient.linear(..color.map.viridis), stroke: 0.3pt + luma(150)),
  box(width: W, height: 7pt, {
    place(left, small[$10^(#lo)$])
    place(right, small[$10^(#hi)$])
    place(center, small[eV/Å²])
  }))

// ============================================================================
// (d) the whole Hessian: hop pattern above the diagonal, magnitudes below
// ============================================================================
#let matrix(W) = box(width: W, height: W, {
  place(polygon(fill: rgb(Sm.upper_ground), stroke: none, (0pt, 0pt), (W, 0pt), (W, W)))     // ground of the upper triangle
  place(image("output/build/matrix.png", width: W, height: W))
  place(line(start: (0pt, 0pt), end: (W, W), stroke: 1.4pt + white))
  place(line(start: (0pt, 0pt), end: (W, W), stroke: 0.6pt + ink))
  place(rect(width: W, height: W, stroke: 0.4pt + luma(120)))
  place(dx: 0pt, dy: 4pt, box(width: W - 4pt, align(right, box(fill: rgb(Sm.upper_ground), inset: 2pt, text(7.5pt, fill: ink)[hop count $k$, key in (b)]))))
  place(dx: 4pt, dy: W - 15pt, box(fill: white, inset: 2pt, text(7.5pt, fill: ink)[$norm(Phi_(i j))$, scale in (c)]))
})

// ============================================================================
// layout: the method on top, the real system below
// ============================================================================
#let P = 1.62in


#block(width: 100%, {
  panel("(a)", note: [#box(baseline: 22%, chain(9pt, j => white, bond: 1.4pt + luma(120), edge: 0.5pt + ink)) #h(5pt) #small[toy chain, 7 atoms, one coordinate each, $K = 2$]])[truncated automatic sparse differentiation]
  schematic()
})
#v(8pt)
#grid(columns: (P, P, P), column-gutter: 1fr, row-gutter: 3pt, align: (left + top),
  box(width: P + 0.25in, panel("(b)")[MOF-177: hops from one atom]),
  panel("(c)")[$norm(Phi_(i j))$ from that atom],
  panel("(d)")[Hessian],
  structure(P, P, mode: "hop"),
  structure(P, P, mode: "mag"),
  matrix(P),
  align(center, hoplegend()),
  align(center, colorbar(P * 0.75, S.logmin, S.logmax)),
  align(center, small[atoms ordered by graph adjacency]),
)
