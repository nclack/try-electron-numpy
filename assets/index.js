console.log("in index.js");
console.log(window.iffy);

import "../node_modules/d3/dist/d3.js";
console.log("here", d3);

const plot = (data) => {
  console.log("here");
  const margin = { top: 10, right: 30, bottom: 30, left: 60 },
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

  console.log(width, height);

  const svg = d3
    .select("#plot")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, data.length]).range([0, width]);
  const y = d3.scaleLinear().domain(d3.extent(data)).range([0, height]);

  svg
    .append("g")
    .attr("class", "xaxis")
    .attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(x));

  svg.append("g").attr("class", "yaxis").call(d3.axisLeft(y));

  svg
    .append("path")
    .datum(data)
    .attr("fill", "none")
    .attr("stroke", "steelblue")
    .attr("stroke-width", 1.5)
    .attr(
      "d",
      d3
        .line()
        .x(function (d, i) {
          return x(i);
        })
        .y(function (d) {
          return y(d);
        })
    );

  console.log(svg);
};

// FIXME: multiple calls to onePythonCall doesn't work.
try {
  const data = window.iffy.onePythonCall("data.f64");
  console.log(data);
  plot(data);
} catch (e) {
  console.log(e);
}
