/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller.integration;

import static org.edgexfoundry.test.data.ProfileData.TEST_LABELS;
import static org.edgexfoundry.test.data.ProfileData.TEST_MAUFACTURER;
import static org.edgexfoundry.test.data.ProfileData.TEST_MODEL;
import static org.edgexfoundry.test.data.ProfileData.TEST_PROFILE_NAME;
import static org.edgexfoundry.test.data.ProfileData.checkTestData;
import static org.edgexfoundry.test.data.ProfileData.newTestInstance;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.controller.impl.DeviceProfileControllerImpl;
import org.edgexfoundry.dao.CommandRepository;
import org.edgexfoundry.dao.DeviceProfileDao;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.dao.DeviceRepository;
import org.edgexfoundry.dao.ProvisionWatcherRepository;
import org.edgexfoundry.domain.meta.Command;
import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.ProvisionWatcher;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.CommandData;
import org.edgexfoundry.test.data.CommonData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ProfileData;
import org.edgexfoundry.test.data.ProvisionWatcherData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;
import org.springframework.web.multipart.MultipartFile;
import org.yaml.snakeyaml.Yaml;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class DeviceProfileControllerTest {

  private static final String LIMIT = "maxLimit";

  @Autowired
  DeviceProfileRepository repos;

  @Autowired
  DeviceProfileControllerImpl controller;

  @Autowired
  CommandRepository cmdRepos;

  @Autowired
  ProvisionWatcherRepository watcherRepos;

  @Autowired
  DeviceRepository deviceRepos;

  @Autowired
  DeviceProfileDao dao;

  private String id;

  // TODO - add test data for resources and deviceResources

  @Before
  public void setup() {
    Command cmd = CommandData.newTestInstance();
    cmdRepos.save(cmd);
    DeviceProfile profile = newTestInstance();
    profile.addCommand(cmd);
    repos.save(profile);
    id = profile.getId();
  }

  @After
  public void cleanup() throws Exception {
    watcherRepos.deleteAll();
    deviceRepos.deleteAll();
    resetControllerMAXLIMIT();
    resetRepos();
    resetDao();
    cmdRepos.deleteAll();
    repos.deleteAll();
  }

  @Test
  public void testDeviceProfile() {
    DeviceProfile device = controller.deviceProfile(id);
    checkTestData(device, id);
  }

  @Test
  public void testDeviceProfileAsYaml() {
    String yaml = controller.deviceProfileAsYaml(id);
    assertTrue("yaml content seems small", yaml.length() > 100);
    // TODO someday make some comparisons on the actual content
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceProfileWithUnknownId() {
    controller.deviceProfile("nosuchid");
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceProfileAsYamlWithUnknownId() {
    controller.deviceProfileAsYaml("nosuchid");
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfileException() throws Exception {
    unsetRepos();
    controller.deviceProfile(id);
  }

  @Test
  public void testDeviceProfiles() {
    List<DeviceProfile> profiles = controller.deviceProfiles();
    assertEquals("Find all not returning a list with one device profile", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesException() throws Exception {
    unsetRepos();
    controller.deviceProfiles();
  }

  @Test(expected = LimitExceededException.class)
  public void testDeviceProfilesMaxLimitExceeded() throws Exception {
    unsetControllerMAXLIMIT();
    controller.deviceProfiles();
  }

  @Test
  public void testDeviceProfileForName() {
    DeviceProfile profile = controller.deviceProfileForName(TEST_PROFILE_NAME);
    checkTestData(profile, id);
  }

  @Test
  public void testDeviceProfileAsYamlForName() {
    String yaml = controller.deviceProfileAsYamlForName(TEST_PROFILE_NAME);
    assertTrue("yaml content seems small", yaml.length() > 100);
    // TODO - someday make some asserts on the actual YAML content
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceProfileForNameWithNoneMatching() {
    controller.deviceProfileForName("badname");
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceProfileAsYamlForNameWithNoneMatching() {
    controller.deviceProfileAsYamlForName("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfileForNameException() throws Exception {
    unsetDao();
    controller.deviceProfileForName(TEST_PROFILE_NAME);
  }

  @Test
  public void testDeviceProfilesByModel() {
    List<DeviceProfile> profiles = controller.deviceProfilesByModel(TEST_MODEL);
    assertEquals("Find for model not returning appropriate list", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test
  public void testDeviceProfilesByModelWithNoneMatching() {
    assertTrue("No devices should be found with bad model",
        controller.deviceProfilesByModel("badmodel").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesByModelException() throws Exception {
    unsetRepos();
    controller.deviceProfilesByModel(TEST_MODEL);
  }

  @Test
  public void testDeviceProfilesByManufacturer() {
    List<DeviceProfile> profiles = controller.deviceProfilesByManufacturer(TEST_MAUFACTURER);
    assertEquals("Find for manufacturer not returning appropriate list", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test
  public void testDeviceProfilesByManufacturerWithNoneMatching() {
    assertTrue("No devices should be found with bad manufacturer",
        controller.deviceProfilesByManufacturer("badmanufacturer").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesByManufacturerException() throws Exception {
    unsetRepos();
    controller.deviceProfilesByManufacturer(TEST_MAUFACTURER);
  }

  @Test
  public void testDeviceProfileByModelOrManufacturer() {
    List<DeviceProfile> profiles =
        controller.deviceProfilesByManufacturerOrModel(TEST_MAUFACTURER, null);
    assertEquals("Find for manufacturer not returning appropriate list", 1, profiles.size());
    checkTestData(profiles.get(0), id);
    controller.deviceProfilesByManufacturerOrModel(null, TEST_MODEL);
    assertEquals("Find for model not returning appropriate list", 1, profiles.size());
    checkTestData(profiles.get(0), id);

  }

  @Test
  public void testDeviceProfilesByManufacturerOrModelWithNoneMatching() {
    assertTrue("No devices should be found with bad manufacturer and model",
        controller.deviceProfilesByManufacturerOrModel("badmanufacturer", "badmodel").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesByManufacturerOrModelException() throws Exception {
    unsetRepos();
    controller.deviceProfilesByManufacturerOrModel(TEST_MAUFACTURER, TEST_MODEL);
  }

  @Test
  public void testDeviceProfileByLabel() {
    List<DeviceProfile> profiles = controller.deviceProfilesByLabel(TEST_LABELS[0]);
    assertEquals("Find for labels not returning appropriate list", 1, profiles.size());
    checkTestData(profiles.get(0), id);
  }

  @Test
  public void testDeviceProfilesByLabelWithNoneMatching() {
    assertTrue("No devices should be found with bad label",
        controller.deviceProfilesByLabel("badlabel").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesByLabelException() throws Exception {
    unsetRepos();
    controller.deviceProfilesByLabel(TEST_LABELS[0]);
  }

  @Test
  public void testAdd() {
    DeviceProfile profile = newTestInstance();
    profile.setName("NewName");
    String newId = controller.add(profile);
    assertNotNull("New device id is null", newId);
    assertNotNull("Modified date is null", profile.getModified());
    assertNotNull("Create date is null", profile.getCreated());
  }

  // TODO - add device resources and resources to test YAML
  @Test
  public void testUploadYamlFile() throws IOException {
    File file = new File("src/test/resources/testprofile.yaml");
    FileInputStream input = new FileInputStream(file);
    try {
      MultipartFile multipartFile = new MockMultipartFile("file", input);
      String id2 = controller.uploadYamlFile(multipartFile);
      assertNotNull("no instance created with Yaml upload", id2);
      DeviceProfile profile2 = controller.deviceProfile(id2);
      assertEquals("Device profile id does not match expected", id2, profile2.getId());
      assertEquals("Device profile name does not match expected", "Test YAML Profile",
          profile2.getName());
      assertEquals("Device profile origin does not match expected", CommonData.TEST_ORIGIN,
          profile2.getOrigin());
      assertEquals("Device profile description does not match expected",
          ProfileData.TEST_DESCRIPTION, profile2.getDescription());
      assertArrayEquals("Device profile labels does not match expected", TEST_LABELS,
          profile2.getLabels());
      assertEquals("Device profile manufacturer does not match expected", TEST_MAUFACTURER,
          profile2.getManufacturer());
      assertEquals("Device profile model does not match expected", TEST_MODEL, profile2.getModel());
      assertNotNull("Device profile modified date is null", profile2.getModified());
      assertNotNull("Device profile created date is null", profile2.getCreated());
    } finally {
      input.close();
    }
  }

  // TODO - add device resources and resources to test YAML
  @Test
  public void testUploadYaml() {
    DeviceProfile profile = newTestInstance();
    profile.setName("newname");
    Yaml yaml = new Yaml();
    String dYaml = yaml.dump(profile);
    String id2 = controller.uploadYaml(dYaml);
    assertNotNull("no instance created with Yaml upload", id2);
    DeviceProfile profile2 = controller.deviceProfile(id2);
    assertEquals("Device profile id does not match expected", id2, profile2.getId());
    assertEquals("Device profile name does not match expected", "newname", profile2.getName());
    assertEquals("Device profile origin does not match expected", CommonData.TEST_ORIGIN,
        profile2.getOrigin());
    assertEquals("Device profile description does not match expected", ProfileData.TEST_DESCRIPTION,
        profile2.getDescription());
    assertArrayEquals("Device profile labels does not match expected", TEST_LABELS,
        profile2.getLabels());
    assertEquals("Device profile manufacturer does not match expected", TEST_MAUFACTURER,
        profile2.getManufacturer());
    assertEquals("Device profile model does not match expected", TEST_MODEL, profile2.getModel());
    assertEquals("Device profile object does not match expected", ProfileData.TEST_OBJ,
        profile2.getObjects());
    assertNotNull("Device profile modified date is null", profile2.getModified());
    assertNotNull("Device profile created date is null", profile2.getCreated());
  }

  @Test(expected = ServiceException.class)
  public void testAddNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithSameName() {
    DeviceProfile profile = newTestInstance();
    controller.add(profile);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() throws Exception {
    unsetRepos();
    DeviceProfile profile = newTestInstance();
    profile.setName("NewName");
    controller.add(profile);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithDuplicateCommandNames() {
    Command cmd1 = CommandData.newTestInstance();
    Command cmd2 = CommandData.newTestInstance();
    DeviceProfile profile = newTestInstance();
    profile.setName("NewName");
    profile.addCommand(cmd1);
    profile.addCommand(cmd2);
    controller.add(profile);
  }

  @Test
  public void testDelete() {
    assertTrue("Delete did not return correctly", controller.delete(id));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteWithNone() {
    controller.delete("badid");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteException() throws Exception {
    unsetRepos();
    controller.delete(id);
  }

  @Test
  public void testDeleteByName() {
    assertTrue("Delete did not return correctly", controller.deleteByName(TEST_PROFILE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameWithNone() {
    controller.delete("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() throws Exception {
    unsetRepos();
    controller.deleteByName(TEST_PROFILE_NAME);
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteWithAssocaitedDevice() {
    DeviceProfile profile = repos.findOne(id);
    ProvisionWatcher watcher = ProvisionWatcherData.newTestInstance();
    watcher.setProfile(profile);
    watcherRepos.save(watcher);
    controller.delete(id);
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteWithAssociateProvisionWatcher() {
    DeviceProfile profile = repos.findOne(id);
    Device device = DeviceData.newTestInstance();
    device.setProfile(profile);
    deviceRepos.save(device);
    controller.delete(id);
  }

  @Test
  public void testUpdate() {
    DeviceProfile profile = repos.findOne(id);
    profile.setDescription("newdescription");
    assertTrue("Update did not complete successfully", controller.update(profile));
    DeviceProfile profile2 = repos.findOne(id);
    assertEquals("Update did not work correclty", "newdescription", profile2.getDescription());
    assertNotNull("Modified date is null", profile2.getModified());
    assertNotNull("Create date is null", profile2.getCreated());
    assertTrue("Modified date and create date should be different after update",
        profile2.getModified() != profile2.getCreated());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() throws Exception {
    unsetRepos();
    DeviceProfile profile = repos.findOne(id);
    profile.setDescription("newdescription");
    controller.update(profile);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithNone() {
    DeviceProfile profile = repos.findOne(id);
    profile.setId("badid");
    profile.setName("badname");
    profile.setDescription("newdescription");
    controller.update(profile);
  }

  @Test(expected = DataValidationException.class)
  public void testUpdateWithTwoCommandSameName() {
    DeviceProfile profile = repos.findOne(id);
    assertFalse("Commands cannot be empty for this test", profile.getCommands().isEmpty());
    Command cmd = CommandData.newTestInstance();
    profile.addCommand(cmd);
    controller.update(profile);
  }

  private void unsetDao() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("dao");
    temp.setAccessible(true);
    temp.set(controller, null);
  }

  private void resetDao() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("dao");
    temp.setAccessible(true);
    temp.set(controller, dao);
  }

  private void unsetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, null);
  }

  private void resetRepos() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField("repos");
    temp.setAccessible(true);
    temp.set(controller, repos);
  }

  private void unsetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 0);
  }

  private void resetControllerMAXLIMIT() throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT);
    temp.setAccessible(true);
    temp.set(controller, 1000);
  }
}
