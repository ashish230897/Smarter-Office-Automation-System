var PythonShell = require('python-shell');
var options = {
    text:'json',
scriptPath: './pythonScripts/'
};
PythonShell.run('my_script.py',options,function (err,results) {
  if (err) throw err;
  console.log(results);
  var body = JSON.parse(results);
  console.log(body.a);
  console.log('finished');
});
