<?php
/*
 * buildSession3.php
 *
 * Builds various HTML forms to process a typical scan session.
 * 
 * HISTORY:
 *  042104 Based on buildSession2.php. Display wizard steps on top. Clone existing scan is a link. Defautl form is create a new scan. AJM (antoine@psych.stanford.edu))
 */

require_once("include.php");
$db = login();

if(isset($_REQUEST["studyId"])){
  $studyId = $_REQUEST["studyId"];
}
else{
  echo "<p class=error>Error: Can't get the study id.</p>\n";
  exit;
}
if(isset($_REQUEST["sessionId"])){
  $sessionId = $_REQUEST["sessionId"];
}
else{
  echo "<p class=error>Error: Can't get the session id.</p>\n";
  exit;
}
$selfUrl = "https://".$_SERVER["HTTP_HOST"].$_SERVER["PHP_SELF"]."?studyId=".$studyId."&sessionId=".$sessionId;
$returnURL = $selfUrl;
$msg = "";

if(isset($_REQUEST["d"])) $d = $_REQUEST["d"];
if(isset($_REQUEST["prevTable"])) $prevTable = $_REQUEST["prevTable"];
if(isset($_REQUEST["deleteId"])) $deleteId = $_REQUEST["deleteId"];
else $deleteId = 0;
$extras = array();
if(isset($_REQUEST["returnURL"])) $extras["returnURL"] = $_REQUEST["returnURL"];
//if(isset($_REQUEST["returnIdName"])) $extras["returnIdName"] = $_REQUEST["returnIdName"];
$extras["returnIdName"] = "scanId";
$scanIdSet = false;
if(isset($_REQUEST["defaultDataId"])) $defaultDataId = $_REQUEST["defaultDataId"];
else $defaultDataId = 0;
if(isset($_REQUEST["msg"])) $msg = $_REQUEST["msg"];
else $msg = "";

if(isset($_REQUEST["table"]))
  $table = $_REQUEST["table"];
else{
  $table = "scans";
  $d[$table]['sessionID'] = $sessionId;
  $d[$table]['primaryStudyID'] = $studyId;
  // Create default scan code
  $q = 'SELECT studyCode FROM studies WHERE id='.$studyId;
  if(!$res = mysql_query($q, $db))
    print "\n<p>mrData ERROR: ".mysql_error($db);
  else {
    $row = mysql_fetch_array($res);
	$studyCode = $row[0];
  }
  $q = 'SELECT start FROM sessions WHERE id='.$sessionId;
  if(!$res = mysql_query($q, $db))
    print "\n<p>mrData ERROR: ".mysql_error($db);
  else {
    $row = mysql_fetch_array($res);
	$time = strtotime($row[0]);
	$date = date('ymd', $time);
  }
  // Figure out scan number
  $q = 'SELECT scanNumber FROM scans WHERE sessionID='.$sessionId.' AND primaryStudyID='.$studyId;
  if(!$res = mysql_query($q, $db))
    print "\n<p>mrData ERROR: ".mysql_error($db);
  else {
    if(mysql_num_rows($res)>0){
      $i = 0;
      while ($row = mysql_fetch_array($res)){
	    $scanNumArray[$i]=$row[0];
	    $i++;
	  }
	  $max1 = max($scanNumArray);
	}
	else
	  $max1 = 0;
  }
  if($max1<127)
    $d[$table]['scanNumber'] = $max1 + 1;
  // Finish scan code
  $max1++;
  if($max1<10)
    $scanNum = '0'.$max1;
  else
    $scanNum = $max1;
  $d[$table]['scanCode'] = $studyCode.$date.'-'.$scanNum;
}
if(isset($prevTable) && is_array($prevTable)) $numPendingTables = count($prevTable);
else $numPendingTables = 0;
//echo "ext(returnURL)=".$extras["returnURL"];
//echo "ext(returnIdName)=".$extras["returnIdName"];
if(isset($_REQUEST[$table])){
  // PROCESS SUBMIT
  // If a submit button with a name matching our current table is set, then
  // we think the user wants to submit the data for this table.
  list($success, $msg, $id) = updateRecord($db, $table, $d[$table], $updateId);
  if($success){
    unset($d[$table]);
    if($numPendingTables>0){
      // pop one off the previous table stack.
      $table = array_pop($prevTable);
    }
    if(isset($extras["returnIdName"]) && $numPendingTables==0){
      /*if (strpos(urldecode($extras["returnURL"]),'?')===FALSE) $urlChar = "?";
      //else $urlChar = "&";
      //if(isset($extras["returnIdName"])){
	  //  echo "111";
	  //  header("Location: buildSession3.php".$urlChar.$extras["returnIdName"]."=".$id);
		//}
      else{
	    echo "222";
	    header("Location: buildSession3.php".$urlChar."id=".$id); 
	  }
	  echo "333";
      exit;*/
	  $scanIdSet = true;
	  $scanId = $id;
    }
  }
}elseif(isset($_REQUEST["cancel"])){
  // PROCESS CANCEL
  $msg = "Cancelled insert/update for table $table.";
  unset($d[$table]);
  if($numPendingTables>0){
    // pop one off the previous table stack.
    $table = array_pop($prevTable);
  }
  if(isset($extras["returnURL"]) && $numPendingTables==0){
    header("Location: ".$returnURL);
    exit;
  }
}elseif(isset($_REQUEST["index"])){
  // PROCESS INDEX
  header("Location: index.php");
  exit;
}

foreach($_REQUEST as $key=>$val){
  if($val=="New"){
    if(isset($prevTable) && is_array($prevTable))
      array_push($prevTable, $table);
    else
      $prevTable = array($table);
    $table = $key;
  }
}

if($deleteId!=0){
	if($numPendingTables>0){
		$msg .= " Refusing to process delete- there are pending tables.";
	}else{
		list($success, $m) = deleteRecord($db, $table, $deleteId);
		if(strlen($msg>0)) $msg .= " / ".$m; else $msg = $m;
		// We do a redirect so that the 'deleteId' won't remain in the url.
		// *** RFD: There's got to be a better way!
		header("Location: ".$_SERVER["PHP_SELF"]."?table=".$table."&msg=".urlencode($msg));
		exit;
	}
}

if(isset($_REQUEST["updateId"]))
  $d[$table]['id'] = $_REQUEST["updateId"];
else 
  $d[$table]['id'] = 0;

if(isset($_REQUEST["scanId"])){
  $scanId = $_REQUEST["scanId"];
  $defaultDataId = $scanId;
}
if($scanIdSet){
  header("Location: buildSessionFinish.php?studyId=".$studyId."&sessionId=".$sessionId."&scanId=".$scanId);
}

writeHeader("Building session- build new scan", "secure", $msg);
wizardHeader(3);
$q = "SELECT sessionCode,primaryStudyID FROM sessions WHERE id=".$sessionId;
if (!$res = mysql_query($q, $db)){
  print "\n<p>mrData ERROR: ".mysql_error($db); exit;
}
$array0 = mysql_fetch_row($res);
$sessionTitle = $array0[0];
$studyId = $array0[1];
//echo "resstr[0]=".$resstr[0];
//echo "resstr[1]=".$resstr[1];
$q = "SELECT title FROM studies WHERE id=".$studyId;
if (!$res = mysql_query($q, $db)){
  print "\n<p>mrData ERROR: ".mysql_error($db); exit;
}
$studyTitle = mysql_fetch_row($res);
echo "<p><font size=-.5><b><ul><li>Study:&nbsp;".$studyTitle[0]."</li></b></font>";
echo "<font size=-.5><b><ul><li>Session:&nbsp;".$sessionTitle."</li></ul></ul></b></font></p>";
echo "<p><font size=+1><a href=\"cloneExistingScan.php?table=scans&studyId=".$studyId."&sessionId=".$sessionId."&returnURL=".urlencode($selfUrl);
echo "&returnIdName=scanId\">Clone existing scan</a>\n";
echo "&nbsp;&nbsp; or add a new scan:\n</font></p>";

/*$table = 'scans';
$d[$table]['sessionID'] = $sessionId;
$d[$table]['primaryStudyID'] = $studyId;
*/
echo buildFormFromTable($db, $table, $prevTable, $selfUrl, $d, $extras, $defaultDataId, "buildSession3");
echo "<br>";


echo "<hr>\n";
echo "<p><font size=+1>Scans entered in this session so far...";
$tableText = displayTable($db, $table, $idStr, "WHERE sessionID=".$sessionId." AND primaryStudyID=".$studyId, 0, $displaySummary, "", $studyId,
  0, $sessionId, $sortbyStr, $sortBy, $sortDir, "");
if($tableText!=""){
  echo $tableText;
}else{
  echo "<p class=error>No entries found for sessionid=".$sessionId.".</p>\n";
}
writeFooter('basic');

?>
