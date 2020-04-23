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

import static org.edgexfoundry.test.data.ProvisionWatcherData.KEY1;
import static org.edgexfoundry.test.data.ProvisionWatcherData.NAME;
import static org.edgexfoundry.test.data.ProvisionWatcherData.VAL1;
import static org.edgexfoundry.test.data.ProvisionWatcherData.checkTestData;
import static org.edgexfoundry.test.data.ProvisionWatcherData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.Field;
import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.controller.impl.ProvisionWatcherControllerImpl;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ProvisionWatcherRepository;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.ProvisionWatcher;
import org.edgexfoundry.exception.controller.DataValidationException;
import org.edgexfoundry.exception.controller.LimitExceededException;
import org.edgexfoundry.exception.controller.NotFoundException;
import org.edgexfoundry.exception.controller.ServiceException;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.edgexfoundry.test.data.ProfileData;
import org.edgexfoundry.test.data.ServiceData;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class ProvisionWatcherControllerTest {

  private static final String LIMIT = "maxLimit";

  @Autowired
  private ProvisionWatcherControllerImpl controller;

  @Autowired
  private ProvisionWatcherRepository repos;

  @Autowired
  private DeviceProfileRepository profileRepos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  private String id;
  private String profileId;
  private String serviceId;

  @Before
  public void setup() {
    DeviceProfile profile = ProfileData.newTestInstance();
    profileRepos.save(profile);
    profileId = profile.getId();
    DeviceService service = ServiceData.newTestInstance();
    serviceRepos.save(service);
    serviceId = service.getId();
    ProvisionWatcher watcher = newTestInstance();
    watcher.setProfile(profile);
    watcher.setService(service);
    repos.save(watcher);
    id = watcher.getId();
  }

  @After
  public void cleanup() throws Exception {
    resetRepos();
    resetControllerMAXLIMIT();
    serviceRepos.deleteAll();
    profileRepos.deleteAll();
    repos.deleteAll();
  }

  @Test
  public void testProvisionWatcher() {
    ProvisionWatcher watcher = controller.watcher(id);
    checkTestData(watcher, id);
  }

  @Test(expected = NotFoundException.class)
  public void testProvisionWatcherWithUnknownId() {
    controller.watcher("nosuchid");
  }

  @Test(expected = ServiceException.class)
  public void testProvisionWatcherException() throws Exception {
    unsetRepos();
    controller.watcher(id);
  }

  @Test
  public void testProvisionWatchers() {
    List<ProvisionWatcher> watchers = controller.watchers();
    assertEquals("Find all not returning a list with one provision watchers", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test(expected = ServiceException.class)
  public void testProvisionWatchersException() throws Exception {
    unsetRepos();
    controller.watchers();
  }

  @Test(expected = LimitExceededException.class)
  public void testProvisionWatchersMaxLimitExceeded() throws Exception {
    unsetControllerMAXLIMIT();
    controller.watchers();
  }

  @Test
  public void testProvisionWatcherForName() {
    ProvisionWatcher watcher = controller.watcherForName(NAME);
    checkTestData(watcher, id);
  }

  @Test(expected = NotFoundException.class)
  public void testProvisionWatcherForNameWithNoneMatching() {
    controller.watcherForName("badname");
  }

  @Test(expected = ServiceException.class)
  public void testProvisionWatcherForNameException() throws Exception {
    unsetRepos();
    controller.watcherForName(NAME);
  }

  @Test
  public void testWatchersForProfile() {
    List<ProvisionWatcher> watchers = controller.watchersForProfile(profileId);
    assertEquals("Find for profiles not returning appropriate list", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test
  public void testWatchersForProfileByName() {
    List<ProvisionWatcher> watchers =
        controller.watchersForProfileByName(ProfileData.TEST_PROFILE_NAME);
    assertEquals("Find for profiles not returning appropriate list", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test(expected = NotFoundException.class)
  public void testWatchersForProfileWithNone() {
    assertTrue("No watchers should be found with bad profile",
        controller.watchersForProfile("badprofile").isEmpty());
  }

  @Test(expected = NotFoundException.class)
  public void testWatchersForProfileByNameWithNone() {
    controller.watchersForProfileByName("badprofile");
  }

  @Test(expected = ServiceException.class)
  public void testWatchersForProfileException() throws Exception {
    unsetRepos();
    controller.watchersForProfile(profileId);
  }

  @Test
  public void testWatchersForService() {
    List<ProvisionWatcher> watchers = controller.watcherForService(serviceId);
    assertEquals("Find for service not returning appropriate list", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test
  public void testWatcherForServiceByName() {
    List<ProvisionWatcher> watchers =
        controller.watcherForServiceByName(ServiceData.TEST_SERVICE_NAME);
    assertEquals("Find for service by name not returning appropriate list", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test(expected = NotFoundException.class)
  public void testWatcherForServiceWithNone() {
    assertNull("No watchers should be found with bad service",
        controller.watcherForService("badserviceId"));
  }

  @Test(expected = NotFoundException.class)
  public void testWatcherForServiceByNameWithNone() {
    assertNull("No watchers should be found with bad service name",
        controller.watcherForServiceByName("badservice"));
  }

  @Test(expected = ServiceException.class)
  public void testWatcherForServiceException() throws Exception {
    unsetRepos();
    controller.watcherForService(serviceId);
  }

  @Test
  public void testWatchersForIdentifier() {
    List<ProvisionWatcher> watchers = controller.watchersForIdentifier(KEY1, VAL1);
    assertEquals("Find for key / value not returning appropriate list", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test
  public void testWatchersForIdentifierWithNone() {
    assertTrue("No watchers should be found with bad key/value identifier pair",
        controller.watchersForIdentifier("badkey", "badvalue").isEmpty());
  }

  @Test(expected = ServiceException.class)
  public void testWatcherForIdentifiersException() throws Exception {
    unsetRepos();
    controller.watchersForIdentifier(KEY1, VAL1);
  }

  @Test
  public void testAdd() {
    DeviceProfile profile = profileRepos.findOne(profileId);
    DeviceService service = serviceRepos.findOne(serviceId);
    ProvisionWatcher watcher = newTestInstance();
    watcher.setName("NewName");
    watcher.setProfile(profile);
    watcher.setService(service);
    String newId = controller.add(watcher);
    assertNotNull("New watcher id is null", newId);
    assertNotNull("Modified date is null", watcher.getModified());
    assertNotNull("Create date is null", watcher.getCreated());
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithSameName() {
    DeviceProfile profile = profileRepos.findOne(profileId);
    DeviceService service = serviceRepos.findOne(serviceId);
    ProvisionWatcher watcher = newTestInstance();
    watcher.setProfile(profile);
    watcher.setService(service);
    controller.add(watcher);
  }

  @Test(expected = ServiceException.class)
  public void testAddException() throws Exception {
    unsetRepos();
    DeviceProfile profile = profileRepos.findOne(profileId);
    DeviceService service = serviceRepos.findOne(serviceId);
    ProvisionWatcher watcher = newTestInstance();
    watcher.setName("NewName");
    watcher.setProfile(profile);
    watcher.setService(service);
    controller.add(watcher);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithNoProfile() throws Exception {
    DeviceService service = serviceRepos.findOne(serviceId);
    ProvisionWatcher watcher = newTestInstance();
    watcher.setName("NewName");
    watcher.setService(service);
    controller.add(watcher);
  }

  @Test(expected = DataValidationException.class)
  public void testAddWithNoService() throws Exception {
    DeviceProfile profile = profileRepos.findOne(profileId);
    ProvisionWatcher watcher = newTestInstance();
    watcher.setName("NewName");
    watcher.setProfile(profile);
    controller.add(watcher);
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
    assertTrue("Delete did not return correctly", controller.deleteByName(NAME));
  }

  @Test(expected = NotFoundException.class)
  public void testDeleteByNameWithNone() {
    controller.delete("badname");
  }

  @Test(expected = ServiceException.class)
  public void testDeleteByNameException() throws Exception {
    unsetRepos();
    controller.deleteByName(NAME);
  }

  @Test
  public void testUpdate() {
    ProvisionWatcher watcher = repos.findOne(id);
    watcher.setOrigin(12345);
    assertTrue("Update did not complete successfully", controller.update(watcher));
    ProvisionWatcher watcher2 = repos.findOne(id);
    assertEquals("Update did not work correclty", 12345, watcher2.getOrigin());
    assertNotNull("Modified date is null", watcher2.getModified());
    assertNotNull("Create date is null", watcher2.getCreated());
    assertTrue("Modified date and create date should be different after update",
        watcher2.getModified() != watcher2.getCreated());
  }

  @Test(expected = ServiceException.class)
  public void testUpdateException() throws Exception {
    unsetRepos();
    ProvisionWatcher watcher = repos.findOne(id);
    watcher.setOrigin(12345);
    controller.update(watcher);
  }

  @Test(expected = NotFoundException.class)
  public void testUpdateWithNone() {
    ProvisionWatcher watcher = repos.findOne(id);
    watcher.setId("badid");
    watcher.setName("badname");
    watcher.setOrigin(12345);
    controller.update(watcher);
  }

  @Test(expected = ServiceException.class)
  public void testUpdateWithNull() {
    controller.update(null);
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
