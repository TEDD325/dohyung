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
 * @microservice: support-logging
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.controller;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.when;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import org.edgexfoundry.controller.impl.CallbackExecutor;
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
import org.edgexfoundry.exception.controller.ClientException;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresNone;
import org.edgexfoundry.test.data.CommandData;
import org.edgexfoundry.test.data.DeviceData;
import org.edgexfoundry.test.data.ProfileData;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.data.domain.Sort;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;
import org.yaml.snakeyaml.DumperOptions.FlowStyle;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.nodes.Tag;

@Category(RequiresNone.class)
public class DeviceProfileControllerTest {

  private static final String LIMIT_PROPERTY = "maxLimit";
  private static final int MAX_LIMIT = 100;

  private static final String TEST_ID = "123";
  private static final String TEST_ERR_MSG = "test message";


  @InjectMocks
  private DeviceProfileControllerImpl controller;

  @Mock
  private DeviceProfileRepository repos;

  @Mock
  private CommandRepository commandRepos;

  @Mock
  private DeviceProfileDao dao;

  @Mock
  private DeviceRepository deviceRepos;

  @Mock
  private ProvisionWatcherRepository watcherRepos;

  @Mock
  private CallbackExecutor callback;

  private DeviceProfile profile;

  @Before
  public void setup() throws Exception {
    MockitoAnnotations.initMocks(this);
    setControllerMAXLIMIT(MAX_LIMIT);
    profile = ProfileData.newTestInstance();
    profile.setId(TEST_ID);
  }

  @Test
  public void testDeviceProfile() {
    when(repos.findOne(TEST_ID)).thenReturn(profile);
    assertEquals("Profile returned is not as expected", profile, controller.deviceProfile(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceProfileNotFound() {
    controller.deviceProfile(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfileException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceProfile(TEST_ID);
  }

  @Test
  public void testDeviceProfiless() {
    List<DeviceProfile> profs = new ArrayList<>();
    profs.add(profile);
    when(repos.findAll(any(Sort.class))).thenReturn(profs);
    when(repos.count()).thenReturn(1L);
    List<DeviceProfile> profiles = controller.deviceProfiles();
    assertEquals("Number of profiles returned does not matched expected number", 1,
        profiles.size());
    assertEquals("Profile returned is not as expected", profile, profiles.get(0));
  }

  @Test(expected = LimitExceededException.class)
  public void testDeviceProfilesMaxLimit() {
    List<DeviceProfile> profs = new ArrayList<>();
    profs.add(profile);
    when(repos.count()).thenReturn(1000L);
    controller.deviceProfiles();
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesException() {
    List<DeviceProfile> profiles = new ArrayList<>();
    profiles.add(profile);
    when(repos.findAll(any(Sort.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
    when(repos.count()).thenReturn(1L);
    controller.deviceProfiles();
  }

  @Test
  public void testDeviceProfileAsYaml() {
    when(repos.findOne(TEST_ID)).thenReturn(profile);
    Yaml yamlProcessor = new Yaml();
    String yaml = yamlProcessor.dumpAs(profile, Tag.MAP, FlowStyle.AUTO);
    assertEquals("Profile YAML returned is not as expected", yaml,
        controller.deviceProfileAsYaml(TEST_ID));
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfileAsYamlException() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceProfileAsYaml(TEST_ID);
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceProfileAsYamlNotFound() {
    assertNull("Unfound profile was not returned as null", controller.deviceProfileAsYaml(TEST_ID));
  }

  @Test
  public void testDeviceProfileForName() {
    when(dao.getByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(profile);
    assertEquals("Device profile returned is not as expected", profile,
        controller.deviceProfileForName(ProfileData.TEST_PROFILE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceProfileForNameNotFound() {
    controller.deviceProfileForName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfileForNameException() {
    when(dao.getByName(ProfileData.TEST_PROFILE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceProfileForName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test
  public void testDeviceProfileByNameAsYaml() {
    when(dao.getByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(profile);
    Yaml yamlProcessor = new Yaml();
    String yaml = yamlProcessor.dumpAs(profile, Tag.MAP, FlowStyle.AUTO);
    assertEquals("Profile YAML returned is not as expected", yaml,
        controller.deviceProfileAsYamlForName(ProfileData.TEST_PROFILE_NAME));
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfileByNameAsYamlException() {
    when(dao.getByName(ProfileData.TEST_PROFILE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceProfileAsYamlForName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test(expected = NotFoundException.class)
  public void testDeviceProfileByNameAsYamlNotFound() {
    controller.deviceProfileAsYamlForName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test
  public void testDeviceProfilesByManufacturer() {
    List<DeviceProfile> profs = new ArrayList<>();
    profs.add(profile);
    when(repos.findByManufacturer(ProfileData.TEST_MAUFACTURER)).thenReturn(profs);
    List<DeviceProfile> profiles =
        controller.deviceProfilesByManufacturer(ProfileData.TEST_MAUFACTURER);
    assertEquals("Number of profiles returned does not matched expected number", 1,
        profiles.size());
    assertEquals("Profile returned is not as expected", profile, profiles.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesByManufacturerException() {
    when(repos.findByManufacturer(ProfileData.TEST_MAUFACTURER))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceProfilesByManufacturer(ProfileData.TEST_MAUFACTURER);
  }

  @Test
  public void testDeviceProfilesByModel() {
    List<DeviceProfile> profs = new ArrayList<>();
    profs.add(profile);
    when(repos.findByModel(ProfileData.TEST_MODEL)).thenReturn(profs);
    List<DeviceProfile> profiles = controller.deviceProfilesByModel(ProfileData.TEST_MODEL);
    assertEquals("Number of profiles returned does not matched expected number", 1,
        profiles.size());
    assertEquals("Profile returned is not as expected", profile, profiles.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesByModelException() {
    when(repos.findByModel(ProfileData.TEST_MODEL)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceProfilesByModel(ProfileData.TEST_MODEL);
  }

  @Test
  public void testDeviceProfilesByManufacturerOrModel() {
    List<DeviceProfile> profs = new ArrayList<>();
    profs.add(profile);
    when(repos.findByManufacturerOrModel(ProfileData.TEST_MAUFACTURER, ProfileData.TEST_MODEL))
        .thenReturn(profs);
    List<DeviceProfile> profiles = controller
        .deviceProfilesByManufacturerOrModel(ProfileData.TEST_MAUFACTURER, ProfileData.TEST_MODEL);
    assertEquals("Number of profiles returned does not matched expected number", 1,
        profiles.size());
    assertEquals("Profile returned is not as expected", profile, profiles.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesByManufacturerOrModelException() {
    when(repos.findByManufacturerOrModel(ProfileData.TEST_MAUFACTURER, ProfileData.TEST_MODEL))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceProfilesByManufacturerOrModel(ProfileData.TEST_MAUFACTURER,
        ProfileData.TEST_MODEL);
  }

  @Test
  public void testDeviceProfilesByLabel() {
    List<DeviceProfile> profs = new ArrayList<>();
    profs.add(profile);
    when(repos.findByLabelsIn(ProfileData.TEST_LABELS[0])).thenReturn(profs);
    List<DeviceProfile> profiles = controller.deviceProfilesByLabel(ProfileData.TEST_LABELS[0]);
    assertEquals("Number of profiles returned does not matched expected number", 1,
        profiles.size());
    assertEquals("Profile returned is not as expected", profile, profiles.get(0));
  }

  @Test(expected = ServiceException.class)
  public void testDeviceProfilesByLabelException() {
    when(repos.findByLabelsIn(ProfileData.TEST_LABELS[0]))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deviceProfilesByLabel(ProfileData.TEST_LABELS[0]);
  }


  @Test
  public void testAdd() {
    when(repos.save(profile)).thenReturn(profile);
    assertEquals("Device Profile ID returned is not the value expected", TEST_ID,
        controller.add(profile));
  }

  @Test(expected = ServiceException.class)
  public void testAddWithNull() {
    controller.add(null);
  }

  @Test(expected = DataValidationException.class)
  public void testAddDuplicateKey() {
    when(repos.save(profile)).thenThrow(new DuplicateKeyException(TEST_ERR_MSG));
    controller.add(profile);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() {
    when(repos.save(profile)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.add(profile);
  }

  @Test
  public void testUploadYmlFile() throws IOException {
    File file = new File("src/test/resources/testprofile.yaml");
    FileInputStream input = new FileInputStream(file);
    try {
      MultipartFile multipartFile = new MockMultipartFile("file", input);
      assertEquals("Device Profile ID returned is not the value expected",
          "57aca2e2555e2e046e6c813d", controller.uploadYamlFile(multipartFile));
    } finally {
      input.close();
    }
  }

  @Test(expected = ClientException.class)
  public void testUploadEmptyYmlFile() throws IOException {
    File file = new File("src/test/resources/emptyprofile.yaml");
    try (FileInputStream input = new FileInputStream(file)) {
      try {
        MultipartFile multipartFile = new MockMultipartFile("file", input);
        controller.uploadYamlFile(multipartFile);
      } finally {
        input.close();
      }
    } finally {
      System.out.println("File src/test/resources/emptyprofile.yaml not provided");
    }
  }

  @Test(expected = ServiceException.class)
  public void testUploadYmlFileException() throws IOException {
    File file = new File("src/test/resources/testprofile.yaml");
    FileInputStream input = new FileInputStream(file);
    try {
      when(repos.save(any(DeviceProfile.class))).thenThrow(new RuntimeException(TEST_ERR_MSG));
      MultipartFile multipartFile = new MockMultipartFile("file", input);
      controller.uploadYamlFile(multipartFile);
    } finally {
      input.close();
    }
  }

  @Test
  public void testUploadYaml() {
    Yaml yaml = new Yaml();
    String yamlString = yaml.dump(profile);
    controller.uploadYaml(yamlString);
    assertEquals("Device Profile ID returned is not the value expected", TEST_ID,
        controller.uploadYaml(yamlString));
  }

  @Test(expected = ServiceException.class)
  public void testUploadYamlException() {
    Yaml yaml = new Yaml();
    String yamlString = yaml.dump(profile);
    controller.uploadYaml(yamlString);
    when(repos.save(profile)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.uploadYaml(yamlString);
  }

  @Test
  public void testUpdate() {
    when(dao.getByIdOrName(profile)).thenReturn(profile);
    assertTrue("Device Profile was not updated", controller.update(profile));
  }

  @Test
  public void testUpdateWithNoProfileID() {
    profile.setId(null);
    when(dao.getByIdOrName(profile)).thenReturn(profile);
    assertTrue("Device Profile was not updated", controller.update(profile));
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithUnknownDeviceProfile() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.update(profile);
  }

  @Test(expected = ServiceException.class)
  public void testUpdatException() {
    when(dao.getByIdOrName(profile)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.update(profile);
  }

  @Test
  public void testUpdateWithCommands() {
    List<Command> cmds = new ArrayList<>();
    cmds.add(CommandData.newTestInstance());
    profile.setCommands(cmds);
    when(dao.getByIdOrName(profile)).thenReturn(profile);
    assertTrue("Device Profile was not updated", controller.update(profile));
  }

  @Test
  public void testDelete() {
    when(repos.findOne(TEST_ID)).thenReturn(profile);
    assertTrue("Device Profile was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteNotFound() {
    when(repos.findOne(TEST_ID)).thenReturn(null);
    controller.delete(TEST_ID);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteDaoFails() {
    when(repos.findOne(TEST_ID)).thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.delete(TEST_ID);
  }

  @Test
  public void testDeleteByName() {
    when(dao.getByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(profile);
    assertTrue("Device Profile was not deleted",
        controller.deleteByName(ProfileData.TEST_PROFILE_NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameNotFound() {
    when(dao.getByName(ProfileData.TEST_PROFILE_NAME)).thenReturn(null);
    controller.deleteByName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameDaoFails() {
    when(dao.getByName(ProfileData.TEST_PROFILE_NAME))
        .thenThrow(new RuntimeException(TEST_ERR_MSG));
    controller.deleteByName(ProfileData.TEST_PROFILE_NAME);
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteWhenAssociatedToDevice() {
    List<Device> devices = new ArrayList<>();
    devices.add(DeviceData.newTestInstance());
    when(deviceRepos.findByProfile(profile)).thenReturn(devices);
    when(repos.findOne(TEST_ID)).thenReturn(profile);
    assertTrue("Device Profile was not deleted", controller.delete(TEST_ID));
  }

  @Test(expected = DataValidationException.class)
  public void testDeleteWhenAssociatedToProvisionWatcher() {
    List<ProvisionWatcher> watchers = new ArrayList<>();
    watchers.add(new ProvisionWatcher());
    when(watcherRepos.findByProfile(profile)).thenReturn(watchers);
    when(repos.findOne(TEST_ID)).thenReturn(profile);
    controller.delete(TEST_ID);
  }

  private void setControllerMAXLIMIT(int newLimit) throws Exception {
    Class<?> controllerClass = controller.getClass();
    Field temp = controllerClass.getDeclaredField(LIMIT_PROPERTY);
    temp.setAccessible(true);
    temp.set(controller, newLimit);
  }

}
