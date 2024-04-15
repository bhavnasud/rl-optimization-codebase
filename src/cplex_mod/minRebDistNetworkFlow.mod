execute
{
  cplex.threads=4;
}
tuple Edge{
  int i;
  int j;
}

tuple edgeAttrTuple{
    int i;
    int j;
    int t;
}

tuple accTuple{
  int i;
  float n;
}

string path = ...;

{edgeAttrTuple} edgeAttr = ...;
{accTuple} accInitTuple = ...;
{accTuple} accRLTuple = ...;

{Edge} edge = {<i,j>|<i,j,t> in edgeAttr};
{int} region = {i|<i,v> in accInitTuple};

float time[edge] = [<i,j>:t|<i,j,t> in edgeAttr];
float desiredCommodity[region] = [i:v|<i,v> in accRLTuple];
float currentCommodity[region] = [i:v|<i,v> in accInitTuple];

dvar int+ rebFlow[edge];
dvar float+ deviation[region];

minimize(sum(e in edge) (rebFlow[e]*time[e]) + 10 * sum(i in region) (deviation[i]));
subject to
{
  forall(i in region)
    {
    abs(currentCommodity[i] - desiredCommodity[i] + sum(e in edge: e.i==i && e.i!=e.j) (rebFlow[<e.j, e.i>] - rebFlow[<e.i, e.j>])) == deviation[i];
    sum(e in edge: e.i==i && e.i!=e.j) rebFlow[<e.i, e.j>] <= currentCommodity[i];
    }
}

main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("flow=[")
  for(var e in thisOplModel.edge)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.rebFlow[e]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.write("deviation=[")
  for(var r in thisOplModel.region)
       {
         ofile.write("(");
         ofile.write(r);
         ofile.write(",");
         ofile.write(thisOplModel.deviation[r]);
         ofile.write(")");
       }
  ofile.writeln("];")
  var obj = cplex.getObjValue();
	ofile.writeln("obj="+obj+";");
  ofile.close();
}