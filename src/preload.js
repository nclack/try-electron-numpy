const {
  doSomethingUseful,
  onePythonCall,
} = require("../build/Release/iffy.node");

// FIXME: Can't load d3 here? It's an ES module.
//        Better seems to be to run a script from index.html and access electron
//        via 'remote' or some other ipc mechanism. See ipcRenderer, ipcMain.
const d3 = require("d3");

const plot = (data) => {
  const margin = { top: 10, right: 30, bottom: 30, left: 60 },
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

  const svg = d3
    .select("plot")
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

  svg.append("path").datum(data).attr("class", "line").attr("d", d3.line());
};

window.addEventListener("DOMContentLoaded", () => {
  const replaceText = (selector, text) => {
    const el = document.getElementById(selector);
    if (el) el.innerText = text;
  };

  for (const dep of ["chrome", "node", "electron"]) {
    replaceText(`${dep}-version`, process.versions[dep]);
  }

  replaceText("hello-world", doSomethingUseful());

  console.log("before python call");
  console.log(plot(onePythonCall("data.f64")));
  console.log("after python call");
});
