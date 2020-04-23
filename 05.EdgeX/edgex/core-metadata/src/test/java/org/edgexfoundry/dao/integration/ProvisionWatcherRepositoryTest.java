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

package org.edgexfoundry.dao.integration;

import static org.edgexfoundry.test.data.ProvisionWatcherData.KEY1;
import static org.edgexfoundry.test.data.ProvisionWatcherData.NAME;
import static org.edgexfoundry.test.data.ProvisionWatcherData.VAL1;
import static org.edgexfoundry.test.data.ProvisionWatcherData.checkTestData;
import static org.edgexfoundry.test.data.ProvisionWatcherData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.DeviceProfileRepository;
import org.edgexfoundry.dao.DeviceServiceRepository;
import org.edgexfoundry.dao.ProvisionWatcherRepository;
import org.edgexfoundry.domain.meta.DeviceProfile;
import org.edgexfoundry.domain.meta.DeviceService;
import org.edgexfoundry.domain.meta.ProvisionWatcher;
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
public class ProvisionWatcherRepositoryTest {

  @Autowired
  private ProvisionWatcherRepository repos;

  @Autowired
  private DeviceProfileRepository profileRepos;

  @Autowired
  private DeviceServiceRepository serviceRepos;

  private String id;
  private String profileId;
  private String serviceId;

  /**
   * Create and save an instance of the ProvisionWatcher before each test Note: the before method
   * tests the save operation of the Repository
   */
  @Before
  public void creatTestData() {
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
    assertNotNull("new test ProvisionWatcher has no identifier", id);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
    profileRepos.deleteAll();
    serviceRepos.deleteAll();
  }

  @Test
  public void testFindOne() {
    ProvisionWatcher watcher = repos.findOne(id);
    assertNotNull("Find one returns no watcher", watcher);
    checkTestData(watcher, id);
  }

  @Test
  public void testFindOneWithBadId() {
    ProvisionWatcher watcher = repos.findOne("badid");
    assertNull("Find one returns device with bad id", watcher);
  }

  @Test
  public void testFindAll() {
    List<ProvisionWatcher> watchers = repos.findAll();
    assertEquals("Find all not returning a list with one watcher", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test
  public void testFindByName() {
    ProvisionWatcher watcher = repos.findByName(NAME);
    assertNotNull("Find by name returns no watcher ", watcher);
    checkTestData(watcher, id);
  }

  @Test
  public void testFindByNameWithBadName() {
    ProvisionWatcher watcher = repos.findByName("badname");
    assertNull("Find by name returns watcher with bad name", watcher);
  }

  @Test
  public void testFindByProfile() {
    List<ProvisionWatcher> watchers = repos.findByProfile(profileRepos.findOne(profileId));
    assertEquals("Find by profile returned no watchers", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test
  public void testFindByProfileWithBadProfile() {
    DeviceProfile profile = new DeviceProfile();
    profile.setId("abc");
    List<ProvisionWatcher> watchers = repos.findByProfile(profile);
    assertTrue("Find by profile returns watchers with bad profile", watchers.isEmpty());
  }

  @Test
  public void testFindByDeviceService() {
    List<ProvisionWatcher> watchers = repos.findByService(serviceRepos.findOne(serviceId));
    assertEquals("No provision watchers found for the assocaited device service", 1,
        watchers.size());
    checkTestData(watchers.get(0), id);
  }

  @Test
  public void testFindByDeviceServiceWithBadService() {
    DeviceService service = new DeviceService();
    service.setId("abcd123");
    assertTrue("Found a watcher by device service with bad device service",
        repos.findByService(service).isEmpty());
  }

  @Test
  public void testFindByIdentifierKeyValue() {
    List<ProvisionWatcher> watchers = repos.findByIdendifierKeyValue("identifiers." + KEY1, VAL1);
    assertEquals("No provision watchers found for the key value pair", 1, watchers.size());
    checkTestData(watchers.get(0), id);
  }
}
