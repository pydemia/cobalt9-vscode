/*
 * Author: Youngju Jaden Kim (pydemia@gmail.com)
 * File Created: 2020-06-30
 * ------------
 * Description: JParser 단위 테스트
 */
package com.example.springbootdemo;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.example.springbootdemo.common.utils.JParser;
import org.junit.Test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class JParserTest {

    private String yamlJsonStr = "{\"name\": \"sample\", \"value\": \"test\"}";

    JParser jParser = new JParser();

    @Test(JParser.class)
    public void readJsonFileTest() throws FileNotFoundException {

        String jsonFilePath = getClass().getResource("/json/explainer-config.json").getPath();

        System.out.println("\n----------- Test: readJsonFile ----------");
        Map<String, Object> jsonMap = (Map<String, Object>) jParser.readJsonFile(jsonFilePath);
        System.out.println(jsonMap.toString());
    }

    @Test
    public void compositeJsonKeyValueTest() throws FileNotFoundException {

        String jsonFilePath = getClass().getResource("/json/explainer-config.json").getPath();
        Map<String, Object> jsonMap = (Map<String, Object>) jParser.readJsonFile(jsonFilePath);

        System.out.println("\n----------- Test: compositeJsonKey ----------");
        List<String> keyList = jParser.compositeJsonKey(jsonMap, "");
        System.out.println(keyList.toString());
        Map<String, Object> jsonFlatMap = jParser.compositeJsonKeyValue(jsonMap, "");
        System.out.println(jsonFlatMap.toString());

    }

    @Test
    public void compositeToComplexTest() throws FileNotFoundException, IOException {

        String jsonFilePath = getClass().getResource("/json/explainer-config.json").getPath();
        Map<String, Object> jsonMap = (Map<String, Object>) jParser.readJsonFile(jsonFilePath);

        Map<String, Object> jsonFlatMap = jParser.compositeJsonKeyValue(jsonMap, "");
        System.out.println(jsonFlatMap.containsValue("exist"));

        System.out.println("\n----------- Test 1-1: FlatMap to Nested(LinkedHashMap) ----------");
        LinkedHashMap jsonComplexLHM = jParser.compositeToComplexLHM(jsonFlatMap);
        LinkedHashMap<String, Object> jsonComplexMap = jParser.compositeToComplexMap(jsonFlatMap);
        // System.out.println(jsonComplexLHM.toString());

        System.out.println("\n----------- Test 1-2: FlatMap to Nested(JsonObject) ----------");
        JsonObject jsonComplexTree = (JsonObject) jParser.compositeToComplexTree(jsonFlatMap);
        JsonObject jsonComplex = jParser.compositeToComplex(jsonFlatMap);
        JsonObject aa = (JsonObject) jsonComplex.get("status");
        JsonArray bb = (JsonArray) aa.get("conditions");
        JsonObject cc = (JsonObject) bb.get(1);
        System.out.println(jsonComplex.toString());

        System.out.println("\n----------- Test 2-1: Nested Map to Nested Json String ----------");
        String jsonStr = jParser.complexToJson(jsonComplex);
        System.out.println(jsonStr);
        String jStr = jParser.complexMapToJson(jsonComplexMap);

        System.out.println("\n----------- Test 2-2: Nested Map to Nested Yaml String ----------");
        String yamlStr = jParser.complexToYaml(jsonComplex);
        System.out.println(yamlStr.toString());
        String yStr = jParser.complexMapToJson(jsonComplexMap);

    }

    @Test
    public void compositeToComplexJsonOrYamlStrTest()
            throws FileNotFoundException, JsonProcessingException, IOException {

        String jsonFilePath = getClass().getResource("/json/explainer-config.json").getPath();
        Map<String, Object> jsonMap = (Map<String, Object>) jParser.readJsonFile(jsonFilePath);

        System.out.println("\n----------- Preparation: Read Json File as Map ----------");
        Map<String, Object> jsonNestedMap = jsonMap;

        System.out.println("\n----------- Preparation: Complex Json to FlatMap ----------");
        Map<String, String> jsonFlatMap = jParser.flatten(jsonNestedMap);

        System.out.println("\n----------- Test 1: FlatMap to Nested Json String ----------");
        String jsonStr = jParser.compositeToComplexJsonStr(jsonFlatMap);
        System.out.println(jsonStr);

        System.out.println("\n----------- Test 2: FlatMap to Nested Yaml String ----------");
        String yamlStr = jParser.compositeToComplexYamlStr(jsonFlatMap);
        System.out.println(yamlStr);

    }

    @Test
    public void dumpTest() throws FileNotFoundException, IOException {

        String jsonFilePath = getClass().getResource("/json/explainer-config.json").getPath();
        Map<String, Object> jsonMap = (Map<String, Object>) jParser.readJsonFile(jsonFilePath);

        Map<String, Object> jsonFlatMap = jParser.compositeJsonKeyValue(jsonMap, "");

        JsonObject jsonComplex = jParser.compositeToComplex(jsonFlatMap);
        LinkedHashMap<String, Object> jsonComplexMap = jParser.compositeToComplexMap(jsonFlatMap);

        System.out.println("\n----------- Test: complexToJson ----------");
        String json = jParser.complexToJson(jsonComplex);
        System.out.println(json);
        String jStr = jParser.complexMapToJson(jsonComplexMap);

        System.out.println("\n----------- Test: complexToJson ----------");
        String yaml = jParser.complexToYaml(jsonComplex);
        System.out.println(yaml);
        String yStr = jParser.complexMapToYaml(jsonComplexMap);

        System.out.println("\n----------- Test: dumpAsJson ----------");
        String jsonSaveFilePath = "/tmp/dump_test.json";
        jParser.dumpAsJson(jsonComplex, jsonSaveFilePath);

        System.out.println("\n----------- Test: dumpAsYaml ----------");
        String yamlSaveFilePath = "/tmp/dump_test.yaml";
        jParser.dumpAsYaml(jsonComplex, yamlSaveFilePath);

        System.out.println("\n----------- Test: Done. ----------");

    }

    @Test
    public void toYamlWithJinjaTest() throws FileNotFoundException, IOException {

        /*
         * System.out.println("\n----------- Test: toYamlWithJinja ----------"); // Get
         * Map from Json // String jsonFilePath =
         * getClass().getResource("/json/explainer-config.json").getPath(); String
         * jsonFilePath = getClass().getResource("/json/mobnet-deploy.json").getPath();
         * Map<String, Object> jsonMap = (Map<String, Object>)
         * jParser.readJsonFile(jsonFilePath);
         * 
         * // Nested to Flattened // Map<String, Object> jsonObjFlatMap = (Map<String,
         * Object>) jParser.compositeJsonKeyValue(jsonMap, ""); Map<String, String>
         * jsonFlatMap = jParser.flatten(jsonMap); System.out.println(jsonFlatMap);
         * 
         * // Create a yaml Jinjava jinjava = new Jinjava(); String jinjaTemplatePath =
         * getClass().getResource("/yaml/jinja-template-comp-key.yaml").getPath();
         * 
         * byte[] encoded = Files.readAllBytes(Paths.get(jinjaTemplatePath)); String
         * template = new String(encoded, Charsets.UTF_8); String yamlOutput =
         * jinjava.render(template, jsonFlatMap); System.out.println(yamlOutput);
         */

    }

    // @Test
    // public void main(String[] args) throws FileNotFoundException {
    public void run() throws Exception {

        this.readJsonFileTest();
        this.compositeJsonKeyValueTest();
        this.compositeToComplexTest();
        this.compositeToComplexJsonOrYamlStrTest();
        this.dumpTest();

    }

    public static void main(String[] args) throws Exception {
        new JParserTest().run();
    }
}